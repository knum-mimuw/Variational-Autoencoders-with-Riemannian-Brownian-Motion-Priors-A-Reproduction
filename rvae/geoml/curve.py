#!/usr/bin/env python3
import torch


class BasicCurve:
    def plot(self, t0=0, t1=1, N=100):
        with torch.no_grad():
            import torchplot as plt

            t = torch.linspace(t0, t1, N)
            points = self(t)  # NxD or BxNxD
            if len(points.shape) == 2:
                points.unsqueeze_(0)  # 1xNxD
            if points.shape[-1] == 1:
                for b in range(points.shape[0]):
                    plt.plot(t, points[b])
            elif points.shape[-1] == 2:
                for b in range(points.shape[0]):
                    plt.plot(points[b, :, 0], points[b, :, 1], "-")
            else:
                print("BasicCurve.plot: plotting is only supported in 1D and 2D")

    def euclidean_length(self, t0=0, t1=1, N=100):
        t = torch.linspace(t0, t1, N)
        points = self(t)  # NxD or BxNxD
        is_batched = points.dim() > 2
        if not is_batched:
            points = points.unsqueeze(0)
        delta = points[:, 1:] - points[:, :-1]  # Bx(N-1)xD
        energies = (delta**2).sum(dim=2)  # Bx(N-1)
        lengths = energies.sqrt().sum(dim=1)  # B
        return lengths


class CubicSpline(BasicCurve):
    def __init__(
        self, begin, end, num_nodes=5, basis=None, device=None, requires_grad=True
    ):
        self.device = device
        # begin # D or 1xD or BxD
        if begin.dim() == 1:
            self.begin = begin.detach().view(1, -1)
        else:
            self.begin = begin.detach()  # BxD

        if end.dim() == 1:
            self.end = end.detach().view(1, -1)
        else:
            self.end = end.detach()

        self.num_nodes = num_nodes
        if basis is None:
            self.basis = self.compute_basis(
                num_edges=num_nodes - 1
            )  # (num_coeffs)x(intr_dim)
        else:
            self.basis = basis
        self.parameters = torch.zeros(
            self.begin.shape[0],
            self.basis.shape[1],
            self.begin.shape[1],
            dtype=self.begin.dtype,
            device=device,
            requires_grad=requires_grad,
        )  # Bx(intr_dim)xD

    # Compute cubic spline basis with end-points (0, 0) and (1, 0)
    def compute_basis(self, num_edges):
        with torch.no_grad():
            # set up constraints
            t = torch.linspace(
                0, 1, num_edges + 1, dtype=self.begin.dtype, device=self.device
            )[1:-1]

            end_points = torch.zeros(
                2, 4 * num_edges, dtype=self.begin.dtype, device=self.device
            )
            end_points[0, 0] = 1.0
            end_points[1, -4:] = 1.0

            zeroth = torch.zeros(
                num_edges - 1, 4 * num_edges, dtype=self.begin.dtype, device=self.device
            )
            for i in range(num_edges - 1):
                si = 4 * i  # start index
                fill = torch.tensor(
                    [1.0, t[i], t[i] ** 2, t[i] ** 3],
                    dtype=self.begin.dtype,
                    device=self.device,
                )
                zeroth[i, si : (si + 4)] = fill
                zeroth[i, (si + 4) : (si + 8)] = -fill

            first = torch.zeros(
                num_edges - 1, 4 * num_edges, dtype=self.begin.dtype, device=self.device
            )
            for i in range(num_edges - 1):
                si = 4 * i  # start index
                fill = torch.tensor(
                    [0.0, 1.0, 2.0 * t[i], 3.0 * t[i] ** 2],
                    dtype=self.begin.dtype,
                    device=self.device,
                )
                first[i, si : (si + 4)] = fill
                first[i, (si + 4) : (si + 8)] = -fill

            second = torch.zeros(
                num_edges - 1, 4 * num_edges, dtype=self.begin.dtype, device=self.device
            )
            for i in range(num_edges - 1):
                si = 4 * i  # start index
                fill = torch.tensor(
                    [0.0, 0.0, 6.0 * t[i], 2.0],
                    dtype=self.begin.dtype,
                    device=self.device,
                )
                second[i, si : (si + 4)] = fill
                second[i, (si + 4) : (si + 8)] = -fill

            constraints = torch.cat((end_points, zeroth, first, second))
            self.constraints = constraints

            ## Compute null space, which forms our basis
            _, S, V = torch.svd(constraints, some=False)
            basis = V[:, S.numel() :]  # (num_coeffs)x(intr_dim)

            return basis

    def __ppeval__(self, t, coeffs):
        # each row of coeffs should be of the form c0, c1, c2, ... representing polynomials
        # of the form c0 + c1*t + c2*t^2 + ...
        # coeffs: Bx(num_edges)x(degree)xD
        B, num_edges, degree, D = coeffs.shape
        idx = (
            torch.floor(t.flatten() * num_edges).clamp(min=0, max=num_edges - 1).long()
        )  # |t| # use this if nodes are equi-distant
        tpow = t.reshape((-1, 1)).pow(
            torch.arange(0.0, degree, device=self.device, dtype=t.dtype).reshape(
                (1, -1)
            )
        )  # |t|x(degree)
        retval = torch.sum(
            tpow.unsqueeze(-1).expand(-1, -1, D).unsqueeze(0) * coeffs[:, idx], dim=2
        )  # Bx|t|xD
        return retval

    def get_coeffs(self):
        coeffs = (
            self.basis.unsqueeze(0)
            .expand(self.parameters.shape[0], -1, -1)
            .bmm(self.parameters)
        )  # Bx(num_coeffs)xD
        B, num_coeffs, D = coeffs.shape
        degree = 4
        num_edges = num_coeffs // degree
        coeffs = coeffs.reshape(B, num_edges, degree, D)  # (num_edges)x4xD
        return coeffs

    def __call__(self, t):
        coeffs = self.get_coeffs()  # Bx(num_edges)x4xD
        retval = self.__ppeval__(t, coeffs)  # Bx|t|xD
        tt = t.reshape((-1, 1)).unsqueeze(0).expand(retval.shape[0], -1, -1)  # Bx|t|x1
        retval += (1 - tt).bmm(self.begin.unsqueeze(1)) + tt.bmm(
            self.end.unsqueeze(1)
        )  # Bx|t|xD
        if (
            retval.shape[0] is 1
        ):  # drop batching if we only have one element in the batch. XXX: This should probably be dropped in the future!
            retval.squeeze_(0)  # |t|xD
        return retval

    def deriv(self, t):
        coeffs = self.get_coeffs()  # Bx(num_edges)x4xD
        B, num_edges, degree, D = coeffs.shape
        dcoeffs = coeffs[:, :, 1:, :] * torch.arange(
            1.0, degree, dtype=t.dtype, device=self.device
        ).reshape(1, 1, -1, 1).expand(
            B, num_edges, -1, D
        )  # Bx(num_edges)x3xD
        retval = self.__ppeval__(t, dcoeffs)  # Bx|t|xD
        # tt = t.reshape((-1, 1)) # |t|x1
        delta = (self.end - self.begin).unsqueeze(1)  # Bx1xD
        retval += delta
        if B is 1:
            retval.unsqueeze_(
                0
            )  # drop batching if we only have one element in the batch. XXX: This should probably be dropped in the future!
        return retval

        # d + c*t + b*t^2 + a*t^3   =>
        # c + 2*b*t + 3*a*t^2


class LinearSpline(BasicCurve):
    def __init__(self, begin, end, num_nodes=5, device=None, requires_grad=True):
        self.device = device if device is not None else begin.device

        # Ensure begin and end are in B×D shape
        self.begin = begin.detach().view(1, -1) if begin.dim() == 1 else begin.detach()
        self.end = end.detach().view(1, -1) if end.dim() == 1 else end.detach()

        self.num_nodes = num_nodes
        B, D = self.begin.shape

        # Only store intermediate nodes (excluding first and last)
        self.parameters = torch.zeros(
            B,
            num_nodes - 2,  # Only intermediate nodes
            D,
            dtype=self.begin.dtype,
            device=self.device,
            requires_grad=requires_grad,
        )

        # Initialize intermediate nodes linearly between begin and end
        with torch.no_grad():
            for i in range(1, num_nodes - 1):
                alpha = i / (num_nodes - 1)
                self.parameters[:, i - 1, :] = (
                    1 - alpha
                ) * self.begin + alpha * self.end

        # Store node positions in [0, 1] range
        self.t_nodes = torch.linspace(
            0, 1, num_nodes, device=self.device, dtype=self.begin.dtype
        )

    def _get_nodes(self):
        """Combine fixed and learnable nodes into full set of nodes"""
        return torch.cat(
            [
                self.begin.unsqueeze(1),  # shape (B, 1, D)
                self.parameters,  # shape (B, num_nodes-2, D)
                self.end.unsqueeze(1),  # shape (B, 1, D)
            ],
            dim=1,
        )  # shape (B, num_nodes, D)

    def __call__(self, t):
        t = t.to(self.device).view(-1)  # shape (T,)

        # Get all nodes (including fixed begin/end)
        all_nodes = self._get_nodes()  # shape (B, num_nodes, D)

        # Find which segment each t falls into
        indices = torch.clamp(
            torch.searchsorted(self.t_nodes, t) - 1,
            0,
            self.num_nodes - 2,
        )  # shape (T,)

        # Get segment boundaries
        t1 = self.t_nodes[indices]  # shape (T,)
        t2 = self.t_nodes[indices + 1]  # shape (T,)

        # Get corresponding nodes
        v1 = all_nodes[0, indices]  # shape (T, D)
        v2 = all_nodes[0, indices + 1]  # shape (T, D)

        # Compute interpolation weights
        w = ((t - t1) / (t2 - t1)).unsqueeze(-1)  # shape (T, 1)

        # Linear interpolation
        retval = (1 - w) * v1 + w * v2  # shape (T, D)

        return retval.squeeze(0) if retval.shape[0] == 1 else retval

    def deriv(self, t):
        t = t.to(self.device).view(-1)  # shape (T,)

        # Get all nodes (including fixed begin/end)
        all_nodes = self._get_nodes()  # shape (B, num_nodes, D)

        # Find which segment each t falls into
        indices = torch.clamp(
            torch.searchsorted(self.t_nodes, t) - 1,
            0,
            self.num_nodes - 2,
        )  # shape (T,)

        # Get segment boundaries
        t1 = self.t_nodes[indices]  # shape (T,)
        t2 = self.t_nodes[indices + 1]  # shape (T,)

        # Get corresponding nodes
        v1 = all_nodes[0, indices]  # shape (T, D)
        v2 = all_nodes[0, indices + 1]  # shape (T, D)

        # Compute derivative (constant within each segment)
        deriv_val = (v2 - v1) / (t2 - t1).unsqueeze(-1)  # shape (T, D)

        return deriv_val.squeeze(0) if deriv_val.shape[0] == 1 else deriv_val


# class LinearSpline(BasicCurve):
#     def __init__(self, begin, end, num_nodes=5, device=None, requires_grad=True):
#         self.device = device
#         # begin # D or 1xD or BxD
#         if begin.dim() == 1:
#             self.begin = begin.detach().view(1, -1)
#         else:
#             self.begin = begin.detach()  # BxD

#         if end.dim() == 1:
#             self.end = end.detach().view(1, -1)
#         else:
#             self.end = end.detach()

#         self.num_nodes = num_nodes
#         self.parameters = torch.zeros(
#             self.begin.shape[0],
#             num_nodes,
#             self.begin.shape[1],
#             dtype=self.begin.dtype,
#             device=device,
#             requires_grad=requires_grad,
#         )

#     def __call__(self, t):
#         """
#         Evaluate the linear spline at given parameter values t.
#         """
#         # Ensure t is a tensor
#         t = t.to(self.device).view(-1, 1)  # |t|x1

#         # Compute indices of the two closest nodes for each t
#         indices = torch.clamp(
#             torch.searchsorted(
#                 torch.linspace(0, 1, self.num_nodes, device=self.device), t
#             )
#             - 1,
#             0,
#             self.num_nodes - 2,
#         )
#         t1 = indices / (self.num_nodes - 1)
#         t2 = (indices + 1) / (self.num_nodes - 1)

#         # Linear interpolation
#         v1 = self.parameters[:, indices, :]  # Bx|t|xD
#         v2 = self.parameters[:, indices + 1, :]  # Bx|t|xD
#         w = (t - t1) / (t2 - t1)  # |t|x1
#         return (1 - w) * v1 + w * v2

#     def deriv(self, t):
#         """
#         Compute the derivative of the linear spline at given parameter values t.
#         """
#         # Ensure t is a tensor
#         t = t.to(self.device).view(-1, 1)  # |t|x1

#         # Compute indices of the two closest nodes for each t
#         indices = torch.clamp(
#             torch.searchsorted(
#                 torch.linspace(0, 1, self.num_nodes, device=self.device), t
#             )
#             - 1,
#             0,
#             self.num_nodes - 2,
#         )
#         t1 = indices / (self.num_nodes - 1)
#         t2 = (indices + 1) / (self.num_nodes - 1)

#         # Compute derivative
#         v1 = self.parameters[:, indices, :]  # Bx|t|xD
#         v2 = self.parameters[:, indices + 1, :]  # Bx|t|xD
#         delta = v2 - v1  # Bx|t|xD
#         return delta / (t2 - t1)
