import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rvae.misc import connecting_geodesic, linear_interpolation
from torchvision.utils import save_image
import plotly.express as px
import plotly.graph_objects as go
from torchvision.utils import make_grid
from tqdm import tqdm


def compute_metric_measure(model, side, step, device, log_scale=True):
    """
    Computes a grid over latent space, evaluates the model metric on each grid point,
    and returns grid axes and the corresponding measure.
    """
    x = np.arange(-side, side, step)
    y = np.arange(-side, side, step)
    xx, yy = np.meshgrid(x, y)
    coords = np.stack((xx.flatten(), yy.flatten()), axis=1).astype(np.float32)
    coords = torch.from_numpy(coords).to(device)

    model.eval()
    with torch.no_grad():
        G = model.metric(coords)
        det = G.det().abs()  # absolute determinant for each grid point

    det = det.detach().cpu().numpy()
    measure = 0.5 * np.log(det) if log_scale else np.sqrt(det)
    num_grid = xx.shape[0]
    measure = measure.reshape(num_grid, num_grid)

    return x, y, measure


def extract_latent_samples(model, data_loader):
    """
    Encodes inputs from the data loader to obtain latent representations and labels.
    """
    samples = []
    labels = []
    for i, data in tqdm(enumerate(data_loader), desc="Embedding datapoints"):
        x, y = data
        # Assuming the input images are 28x28 and need to be flattened.
        z, _, _ = model.encode(x.view(-1, 784))
        samples.append(z)
        labels.append(y)
    samples = torch.stack(samples, dim=0).view(-1, 2)
    labels = torch.stack(labels).view(-1)

    idxs = torch.arange(len(labels))
    return samples, labels, idxs


def extract_latent_mu(model, data_loader, model_type: str):
    """
    Encodes inputs from the data loader to obtain latent representations and labels.
    """
    assert model_type in ["VAE", "RVAE"]
    mus = []
    labels = []
    for i, data in tqdm(enumerate(data_loader), desc="Embedding datapoints"):
        x, y = data
        # Assuming the input images are 28x28 and need to be flattened.
        if model_type == "RVAE":
            _, mu, _ = model.encode(x.view(-1, 784))
        elif model_type == "VAE":
            mu, _ = model.encode(x.view(-1, 784))
        # print(f"DEBUG : {mu.shape=}")
        mus.append(mu)
        labels.append(y)

    mus = torch.cat(mus, dim=0)
    # mus = mus.view(-1, 2)
    labels = torch.cat(labels, dim=0)

    idxs = torch.arange(len(labels))
    print(f"DEBUG : {mus.shape=}")
    print(f"DEBUG : {labels.shape=}")
    print(f"DEBUG : {idxs.shape=}")
    return mus, labels, idxs


def compute_geodesic_and_linear(
    model,
    sample1,
    sample2,
    n_nodes=32,
    eval_grid=64,
    max_iter=500,
    l_rate=1e-3,
    is_cubic=True,
):
    """
    Computes the geodesic connecting two latent points along with the corresponding linear interpolation.

    Assumes that helper functions `connecting_geodesic` and `linear_interpolation` are defined.
    """
    # Adjust beta for geodesic computation.
    original_beta = model.p_sigma._modules["0"].beta
    model.p_sigma._modules["0"].beta = 0.07

    # Compute the geodesic path between the two points.
    c, _ = connecting_geodesic(
        model,
        sample1.unsqueeze(0),
        sample2.unsqueeze(0),
        n_nodes=n_nodes,
        eval_grid=eval_grid,
        max_iter=max_iter,
        l_rate=l_rate,
        is_cubic=is_cubic,
    )
    t_values = torch.arange(0, 1, 0.05)
    c_pts = c(t_values).detach().cpu().numpy()

    # Reset beta parameter.
    model.p_sigma._modules["0"].beta = 0.01

    # Compute linear interpolation between the geodesic endpoints.
    lin_pts = linear_interpolation(
        torch.from_numpy(c_pts[0]), torch.from_numpy(c_pts[-1]), n_points=20
    )

    return c_pts, lin_pts


def save_interpolated_images(model, c_pts, lin_pts, save_dir):
    """
    Decodes geodesic and linear interpolation latent points into images and saves them.
    """
    # Decode geodesic path images.
    c_pts_tensor = torch.from_numpy(c_pts).float()
    p_mu, _ = model.decode(c_pts_tensor, False)
    save_image(
        p_mu.view(-1, 1, 28, 28).detach().cpu(), save_dir / "geodesic_img.jpeg", nrow=20
    )
    # Decode linear interpolation images.
    p_mu, _ = model.decode(lin_pts, False)
    save_image(
        p_mu.view(-1, 1, 28, 28).detach().cpu(), save_dir / "linear_img.jpeg", nrow=20
    )


# --- Plotting Functions ---
def create_metric_heatmap(
    model,
    side=110,
    step=1,
    device=None,
    log_scale=True,
    height=800,
    width=800,
):
    # Compute metric background.
    x, y, measure = compute_metric_measure(model, side, step, device, log_scale)

    # Create the heatmap using px.imshow.
    heatmap_fig = px.imshow(
        measure,
        origin="lower",
        x=x,
        y=y,
        color_continuous_scale="RdGy",
        labels={
            "x": "Latent Dimension 1",
            "y": "Latent Dimension 2",
            "color": "Metric",
        },
    )
    # Adjust the heatmap colorbar to avoid collision.
    heatmap_fig.data[0].update(colorbar=dict(title="Metric", x=0.02, len=0.3))

    heatmap_fig.update_layout(
        title=dict(
            text="Latent Space Visualization",
            pad=dict(b=20),  # adds 20 pixels of space below the title
        ),
        margin=dict(
            t=150,  # space at the top (ensures title isn't too cramped)
            b=100,  # space at the bottom of the plot
        ),
        xaxis_title="Latent Dimension 1",
        yaxis_title="Latent Dimension 2",
        width=width,
        height=height,
        legend=dict(
            title=dict(text="Label"),
            orientation="h",
            x=0.5,  # center horizontally
            y=1.0,
            xanchor="center",
            yanchor="bottom",
        ),
    )

    return heatmap_fig


def tensor_to_dataframe(tensor, col_names=["Latent Dimension 1", "Latent Dimension 2"]):
    """
    Converts a PyTorch tensor with shape (N, 2) to a pandas DataFrame.

    Parameters:
        tensor (torch.Tensor): Input tensor of shape (N, 2).
        col_names (list): List of column names.

    Returns:
        pd.DataFrame: A DataFrame with the provided column names.
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    # Detach and convert to numpy array
    np_array = tensor.detach().numpy()
    df = pd.DataFrame(np_array, columns=col_names)
    return df


# def overlay_kde_contours(
#     heatmap_fig,
#     data_tensor,
#     line_color="black",
#     line_width=2,
#     show_points=False,
#     toggle_contours=True,
# ):
#     """
#     Overlays density contour lines (KDE) onto an existing Plotly heatmap figure and adds buttons
#     to toggle their visibility.

#     Parameters:
#         heatmap_fig (plotly.graph_objects.Figure): The existing heatmap figure.
#         data_tensor (Tensor or np.ndarray): Data points (with two columns) to compute the KDE contours.
#         line_color (str): Color for the contour lines.
#         line_width (int): Width of the contour lines.
#         show_points (bool): If True, also overlay the raw data as scatter points.
#         toggle_contours (bool): If True, adds update buttons to show/hide the contour traces.

#     Returns:
#         plotly.graph_objects.Figure: The updated figure with overlayed contours and toggle buttons.
#     """
#     # Convert the tensor to a DataFrame
#     data = tensor_to_dataframe(data_tensor)
#     x_col = "Latent Dimension 1"
#     y_col = "Latent Dimension 2"

#     # Create a density contour figure using Plotly Express.
#     density_fig = px.density_contour(data, x=x_col, y=y_col)

#     # Customize each contour trace and add it to the heatmap.
#     contour_indices = []
#     for trace in density_fig.data:
#         trace.line.color = line_color
#         trace.line.width = line_width
#         trace.showscale = False  # turn off the separate color scale
#         trace.name = "Density Contour"
#         heatmap_fig.add_trace(trace)
#         contour_indices.append(len(heatmap_fig.data) - 1)

#     # Optionally, overlay scatter points for the raw data.
#     if show_points:
#         scatter_fig = px.scatter(data, x=x_col, y=y_col)
#         for trace in scatter_fig.data:
#             heatmap_fig.add_trace(trace)

#     # If toggle_contours is True, add update menu buttons to show/hide the contour traces.
#     if toggle_contours:
#         num_traces = len(heatmap_fig.data)
#         # Start with all traces visible.
#         visible_all = [True] * num_traces
#         # Create a copy where contour traces are hidden.
#         visible_hide = visible_all.copy()
#         for idx in contour_indices:
#             visible_hide[idx] = False

#         heatmap_fig.update_layout(
#             updatemenus=[
#                 dict(
#                     type="buttons",
#                     direction="left",
#                     buttons=list(
#                         [
#                             dict(
#                                 label="Show Contours",
#                                 method="update",
#                                 args=[
#                                     {"visible": visible_all},
#                                     {"title": "Contours Visible"},
#                                 ],
#                             ),
#                             dict(
#                                 label="Hide Contours",
#                                 method="update",
#                                 args=[
#                                     {"visible": visible_hide},
#                                     {"title": "Contours Hidden"},
#                                 ],
#                             ),
#                         ]
#                     ),
#                     pad={"r": 10, "t": 10},
#                     showactive=True,
#                     x=0.0,
#                     xanchor="left",
#                     y=1.1,
#                     yanchor="top",
#                 )
#             ]
#         )

#     return heatmap_fig


def overlay_kde_contours(
    heatmap_fig, data_tensor, line_color="black", line_width=2, show_points=False
):
    data = tensor_to_dataframe(data_tensor)
    # Create a density contour figure using Plotly Express.
    x_col = "Latent Dimension 1"
    y_col = "Latent Dimension 2"
    density_fig = px.density_contour(data, x=x_col, y=y_col)

    # Customize each trace: set the contour line color and width,
    # and hide the color scale for these contours.
    for trace in density_fig.data:
        trace.line.color = line_color
        trace.line.width = line_width
        trace.showscale = False  # turn off the separate color scale

    # Optionally, overlay scatter points for the raw data.
    if show_points:
        scatter_fig = px.scatter(data, x=x_col, y=y_col)
        for trace in scatter_fig.data:
            heatmap_fig.add_trace(trace)

    # Add each density contour trace to the original heatmap.
    for trace in density_fig.data:
        heatmap_fig.add_trace(trace)

    return heatmap_fig


def create_base_plot(
    model,
    data_loader,
    side=110,
    step=1,
    device=None,
    log_scale=True,
    height=800,
    width=800,
):
    """
    Creates and returns a base Plotly figure with:
      - A heatmap of the metric measure computed over a grid in latent space.
      - A scatter plot of latent representations with discrete color coding for each digit.

    This version uses Plotly Express to automatically create discrete traces for the digits,
    so that the legend shows the digits with their colors and no additional colorbar is needed.
    """

    heatmap_fig = create_metric_heatmap(
        model=model, side=side, step=step, device=device, log_scale=log_scale
    )

    # Extract latent representations.
    samples, labels, idxs = extract_latent_samples(model, data_loader)
    samples_np = samples.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # Create a DataFrame for the scatter data.
    df = pd.DataFrame(
        {
            "idx": idxs,
            "x": samples_np[:, 0],
            "y": samples_np[:, 1],
            "label": labels_np.astype(str),
        }
    )
    # Create the scatter plot using Plotly Express (automatically creates discrete traces).
    scatter_fig = px.scatter(
        df,
        x="x",
        y="y",
        color="label",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        hover_data={"idx": True, "label": True, "x": ":.2f", "y": ":.2f"},
        opacity=0.6,
    )
    scatter_fig.update_traces(marker=dict(size=4))

    # Combine the scatter traces with the heatmap without an explicit loop.
    heatmap_fig.add_traces(scatter_fig.data)

    # Update layout.
    heatmap_fig.update_layout(
        title=dict(
            text="Latent Space Visualization",
            pad=dict(b=20),  # adds 20 pixels of space below the title
        ),
        margin=dict(
            t=150,  # space at the top (ensures title isn't too cramped)
            b=100,  # space at the bottom of the plot
        ),
        xaxis_title="Latent Dimension 1",
        yaxis_title="Latent Dimension 2",
        width=width,
        height=height,
        legend=dict(
            title=dict(text="Label"),
            orientation="h",
            x=0.5,  # center horizontally
            y=1.0,
            xanchor="center",
            yanchor="bottom",
        ),
    )

    return heatmap_fig, samples_np, labels_np, idxs


def overlay_bm_samples(
    heatmap_fig, bm_samples, line_color="black", line_width=2, marker_size=1
):
    """
    Overlays Brownian motion samples on a given Plotly heatmap figure.

    Parameters:
        heatmap_fig (plotly.graph_objects.Figure):
            The Plotly heatmap figure.
        bm_samples (np.ndarray or torch.Tensor):
            An array or tensor representing BM sample points. If 3D, assumed to be
            (num_steps, num_samples, 2) and iterated over the 'num_samples' dimension.
            If 2D, assumed to be (num_steps, 2) - a single BM sample trajectory.
        line_color (str, optional):
            Color for the BM line overlay (default: "black").
        line_width (int, optional):
            Width of the BM line (default: 2).
        marker_size (int, optional):
            Size of markers (default: 1).

    Returns:
        plotly.graph_objects.Figure: The heatmap figure with the BM sample overlay.
    """
    # Convert torch tensor to numpy array if necessary.
    if isinstance(bm_samples, torch.Tensor):
        bm_samples = bm_samples.detach().cpu().numpy()

    # Debug: Print the range of BM sample coordinates (if bm_samples is at least 2D)
    if bm_samples.ndim >= 2:
        print(
            "BM samples x-range: [{:.2f}, {:.2f}]".format(
                bm_samples[..., 0].min(), bm_samples[..., 0].max()
            )
        )
        print(
            "BM samples y-range: [{:.2f}, {:.2f}]".format(
                bm_samples[..., 1].min(), bm_samples[..., 1].max()
            )
        )

    # If bm_samples is 3D assume shape (num_steps, num_samples, 2) and iterate over samples.
    if bm_samples.ndim == 3:
        num_samples = bm_samples.shape[1]
        for i in range(num_samples):
            sample = bm_samples[:, i, :]  # shape (num_steps, 2)
            bm_trace = go.Scatter(
                x=sample[:, 0],
                y=sample[:, 1],
                mode="lines+markers",  # show both lines and markers
                marker=dict(size=marker_size),
                line=dict(color=line_color, width=line_width),
                name=f"BM Sample {i}",
                opacity=1,
            )
            heatmap_fig.add_trace(bm_trace)
    # Otherwise, if bm_samples is 2D, add a single trace.
    elif bm_samples.ndim == 2:
        bm_trace = go.Scatter(
            x=bm_samples[:, 0],
            y=bm_samples[:, 1],
            mode="lines+markers",  # show both lines and markers
            marker=dict(size=marker_size),
            line=dict(color=line_color, width=line_width),
            name="BM Sample",
            opacity=1,
        )
        heatmap_fig.add_trace(bm_trace)

    return heatmap_fig


def rgb_to_hex(rgb):
    """Convert an RGB tuple (values between 0 and 1) to a hex string."""
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )


def overlay_bm_samples_gradient(
    heatmap_fig, bm_samples, colormap="hot", line_width=2, marker_size=1, name_sufix=""
):
    """
    Overlays BM sample segments with a gradient color indicating progression.
    Each segment of the BM trajectory gets a color from the colormap (cold for early, warm for later).

    Parameters:
        heatmap_fig (plotly.graph_objects.Figure): The figure to add traces to.
        bm_samples (np.ndarray or torch.Tensor): BM sample points.
            If 3D, assumed shape is (num_steps, num_samples, 2).
            If 2D, assumed shape is (num_steps, 2), which will be converted to (num_steps, 1, 2).
        colormap (str): Name of the matplotlib colormap (default "hot").
        line_width (int): Width of the line segments.
        marker_size (int): Size of the markers.

    Returns:
        plotly.graph_objects.Figure: The figure with the added BM traces.
    """
    # Convert torch tensor to numpy array if necessary.
    if isinstance(bm_samples, torch.Tensor):
        bm_samples = bm_samples.detach().cpu().numpy()

    # If bm_samples is 2D, convert it to 3D with one sample.
    if bm_samples.ndim == 2:
        bm_samples = bm_samples[:, None, :]  # shape becomes (num_steps, 1, 2)

    num_steps, num_samples, _ = bm_samples.shape
    cmap = plt.get_cmap(colormap)

    # Loop over each sample trajectory.
    for sample_idx in range(num_samples):
        sample = bm_samples[:, sample_idx, :]  # shape (num_steps, 2)
        # For each segment, assign it a color from the colormap.
        for step in range(num_steps - 1):
            # Compute the colormap value so that later segments are "warmer".
            col_val = step / (num_steps - 1)
            current_color = rgb_to_hex(cmap(col_val)[:3])
            trace = go.Scatter(
                x=[sample[step, 0], sample[step + 1, 0]],
                y=[sample[step, 1], sample[step + 1, 1]],
                mode="lines+markers",
                marker=dict(size=marker_size, color=current_color),
                line=dict(color=current_color, width=line_width),
                # Only add legend entry once.
                name=(
                    f"BM Sample (gradient) {name_sufix}"
                    if sample_idx == 0 and step == 0
                    else ""
                ),
                showlegend=True if sample_idx == 0 and step == 0 else False,
                opacity=1,
            )
            heatmap_fig.add_trace(trace)

    return heatmap_fig


def overlay_geodesic_on_plot(fig, c_pts, lin_pts):
    """
    Overlays geodesic and linear interpolation curves onto an existing Plotly figure.

    - c_pts: Array of geodesic curve points.
    - lin_pts: Array of linear interpolation points.

    Returns the updated figure.
    """
    # Add geodesic path.
    geodesic_trace = go.Scatter(
        x=c_pts[:, 0],
        y=c_pts[:, 1],
        mode="lines",
        line=dict(color="black"),
        name="Geodesic Path",
    )
    fig.add_trace(geodesic_trace)

    # Add linear interpolation path.
    linear_trace = go.Scatter(
        x=lin_pts[:, 0],
        y=lin_pts[:, 1],
        mode="lines",
        line=dict(color="red"),
        name="Linear Interpolation",
    )
    fig.add_trace(linear_trace)

    # Optionally update the title.
    fig.update_layout(
        title=dict(
            text="Latent Space Visualization with Geodesic Overlay",
        )
    )

    return fig


# --- Main Function ---


def plot_geodesics(model, data_loader, save_dir, device, log_scale=True):
    """
    Main function that:
      1. Generates a base Plotly figure (heatmap + latent scatter).
      2. Randomly selects two latent points to compute the geodesic.
      3. Computes geodesic and linear interpolation paths.
      4. Saves the decoded images of these paths.
      5. Overlays the computed curves on the base plot.

    Returns the geodesic and linear interpolation points.
    """
    # Create the base plot (this part is computed only once).
    fig, samples_np, _ = create_base_plot(
        model, data_loader, side=110, step=1, device=device, log_scale=log_scale
    )

    # Randomly select two latent samples.
    idx = np.random.choice(samples_np.shape[0], 2, replace=False)
    sample1 = torch.from_numpy(samples_np[idx[0]]).to(device)
    sample2 = torch.from_numpy(samples_np[idx[1]]).to(device)

    # Compute geodesic and linear interpolation curves.
    c_pts, lin_pts = compute_geodesic_and_linear(model, sample1, sample2)

    # Save the decoded images for the geodesic and linear interpolation.
    save_interpolated_images(model, c_pts, lin_pts, save_dir)

    # Overlay the geodesic curves on the pre-computed base plot.
    fig = overlay_geodesic_on_plot(fig, c_pts, lin_pts)
    fig.show()

    return c_pts, lin_pts


def get_interpolated_images(model, c_pts, lin_pts, nrow=20):
    """
    Decodes geodesic and linear interpolation latent points into image grids.

    Parameters:
      model  : The autoencoder model.
      c_pts  : A NumPy array of latent points along the geodesic.
      lin_pts: A tensor of latent points for linear interpolation.
      nrow   : Number of images in each row in the grid. Defaults to 20.

    Returns:
      A tuple (geodesic_grid, linear_grid) of image tensors.
    """
    # Decode geodesic path images.
    c_pts_tensor = torch.from_numpy(c_pts).float()
    p_mu, _ = model.decode(c_pts_tensor, False)
    geodesic_grid = make_grid(p_mu.view(-1, 1, 28, 28).detach().cpu(), nrow=nrow)

    # Decode linear interpolation images.
    p_mu, _ = model.decode(lin_pts, False)
    linear_grid = make_grid(p_mu.view(-1, 1, 28, 28).detach().cpu(), nrow=nrow)

    return geodesic_grid, linear_grid


def compare_interpolations(geodesic_grid, linear_grid):
    combined_grid = torch.cat((geodesic_grid, linear_grid), dim=1)

    # Plot the combined grid.
    plt.figure(figsize=(10, 10))
    # For a grayscale image, we use cmap="gray"
    plt.imshow(combined_grid.permute(1, 2, 0).numpy(), cmap="gray")
    plt.axis("off")
    # plt.title("Geodesic (top) and Linear (bottom) Interpolations")
    plt.show()
