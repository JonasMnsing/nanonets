import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# Default Colors
BLUE_COLOR = '#4477AA'
RED_COLOR = '#EE6677'
GREEN_COLOR = '#228833'
ELECTRODE_RADIUS = 10.0

def display_landscape(data_row: np.ndarray, Nx: int, Ny: int, fig=None, ax=None, 
                      cmap='coolwarm', vmin=None, vmax=None,
                      x_label='$x_{NP}$', y_label='$y_{NP}$', colorbar=False, 
                      interpolation=None, cbar_label=''):
    """Displays a 2D landscape of a nanoparticle network from a 1D array."""
    
    # Reshape the 1D slice into the 2D grid
    arr = data_row.reshape(Nx, Ny)
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(layout='constrained')
    
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', interpolation=interpolation)
    ax.set_xticks(np.arange(Nx))
    ax.set_yticks(np.arange(Ny))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if colorbar:
        fig.colorbar(im, ax=ax, label=cbar_label)

    return fig, ax

def animate_landscape(landscape: np.ndarray, Nx: int, Ny: int, fig=None, ax=None, 
                      cmap='coolwarm', vmin=None, vmax=None,
                      x_label='$x_{NP}$', y_label='$y_{NP}$', interpolation=None, 
                      delay_between_frames=200, cbar_width=0.05, cbar_label='', plot_steps=False):
    """Animates a time-series of 2D nanoparticle landscapes."""
    
    N_rows = landscape.shape[0]

    if vmin is None:
        vmin = np.min(landscape)
    if vmax is None:
        vmax = np.max(landscape)
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(layout='constrained')

    # Setup axes and colorbar once
    cax = ax.inset_axes([1.03, 0, cbar_width, 1], transform=ax.transAxes)
    ax.set_xticks(np.arange(Nx))
    ax.set_yticks(np.arange(Ny))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Draw a dummy image to initialize the colorbar
    dummy_im = ax.imshow(landscape[0, :].reshape(Nx, Ny), cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    fig.colorbar(dummy_im, cax=cax, label=cbar_label)

    ims = []
    for i in range(N_rows):
        # We only animate the image and the title, not the colorbar
        im = ax.imshow(landscape[i, :].reshape(Nx, Ny), cmap=cmap, vmin=vmin, vmax=vmax,
                       origin='lower', interpolation=interpolation, animated=True)
        
        frame_artists = [im]
        if plot_steps:
            title = ax.text(0.5, 1.05, f"Step: {i+1}/{N_rows}", 
                            size=plt.rcParams["axes.titlesize"],
                            ha="center", transform=ax.transAxes)
            frame_artists.append(title)
        
        ims.append(frame_artists)

    ani = animation.ArtistAnimation(fig, ims, interval=delay_between_frames, 
                                    repeat_delay=delay_between_frames*10, blit=False)
    return ani

def display_network(G: nx.Graph, pos: dict, radius: np.ndarray, fig=None, ax=None, e_facecolor=None):
    """Visualizes a nanoparticle network, including particles and electrodes."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(layout='constrained')
        
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw network edges
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], 'black', lw=1, alpha=0.3)

    # Draw electrodes
    N_e = len(G.nodes) - len(radius)
    if e_facecolor is None:
        e_facecolor = [RED_COLOR for _ in range(N_e)]
        
    for i in range(1, N_e + 1):
        x, y = pos[-i] # Electrodes are expected at -1, -2, ...
        circ = plt.Circle((x, y), ELECTRODE_RADIUS, fill=True,
                            edgecolor='black', lw=1, zorder=2, facecolor=e_facecolor[i-1])
        ax.add_patch(circ)
    
    # Draw nanoparticles
    for i in range(len(radius)):
        x, y = pos[i] # NPs are expected to be at index 0, 1, ...
        circle = plt.Circle((x, y), radius[i], fill=True,
                            edgecolor='black', lw=1, zorder=2, facecolor=BLUE_COLOR)
        ax.add_patch(circle)

    # Autoscale and padding
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = max(np.max(radius), ELECTRODE_RADIUS) + 1
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    
    return fig, ax

def update_circle_colors(ax: plt.Axes, pot: np.ndarray, vlim=None, cmap="coolwarm", colorbar=False):
    """Updates nanoparticle circles colored by potential."""
    vlim = np.max(np.abs(pot)) if vlim is None else vlim
    norm = Normalize(-vlim, vlim)
    
    # Modern matplotlib colormap call
    cmap_obj = plt.colormaps.get_cmap(cmap)
    
    # We only update the patches that belong to nanoparticles (skipping electrodes if appended last)
    # Assumes ax.patches[i] matches pot[i]
    for i in range(len(pot)):
        ax.patches[i].set_facecolor(cmap_obj(norm(pot[i])))
    
    if colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label='$\phi$ [V]')

def display_network_currents(current_dict: dict, N_electrodes: int, time_step: int = -1, 
                             pot_landscape: np.ndarray = None, pos: dict = None, 
                             fig=None, ax=None, arrowsize=12, node_size=300, 
                             blue_color=BLUE_COLOR, red_color=RED_COLOR,
                             position_by_currents=False, edge_vmin=None, edge_vmax=None):
    """Visualizes the directed net currents inside the nanoparticle network."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(layout='constrained')
        
    ax.axis('off')

    values_new, junctions_new = [], []
    processed_pairs = set()

    # Process the net currents directly from the dictionary
    for (i, j), val_array in current_dict.items():
        if (i, j) in processed_pairs or (j, i) in processed_pairs:
            continue
            
        # Extract scalar values for the given time step
        val1 = val_array[time_step] if isinstance(val_array, np.ndarray) else val_array
        
        # Check for reverse current
        val2 = 0.0
        if (j, i) in current_dict:
            rev_array = current_dict[(j, i)]
            val2 = rev_array[time_step] if isinstance(rev_array, np.ndarray) else rev_array

        net_current = np.abs(val2 - val1)
        values_new.append(net_current)
        
        # Shift indices to match NetworkX pos layout (NPs start at 0, Electrodes at -1, -2...)
        idx_i = int(i) - N_electrodes if int(i) >= N_electrodes else -(int(i) + 1)
        idx_j = int(j) - N_electrodes if int(j) >= N_electrodes else -(int(j) + 1)
        
        if val1 > val2:
            junctions_new.append((idx_i, idx_j))
        else:
            junctions_new.append((idx_j, idx_i))
            
        processed_pairs.add((i, j))

    # Log scaling for visual clarity of orders of magnitude
    values_new = np.log1p(values_new)
    if np.max(values_new) > np.min(values_new):
        values_new = (values_new - np.min(values_new)) / (np.max(values_new) - np.min(values_new))
    else:
        values_new = np.zeros_like(values_new)

    edge_vmin = np.min(values_new) if edge_vmin is None else edge_vmin
    edge_vmax = np.max(values_new) if edge_vmax is None else edge_vmax

    G = nx.DiGraph()
    unique_nodes = set([n for junc in junctions_new for n in junc])
    G.add_nodes_from(unique_nodes)

    if pot_landscape is not None:
        colors = [red_color if (n >= 0 and pot_landscape[n] < 0) else blue_color for n in G.nodes]
        states = [node_size for _ in G.nodes] 
    else:
        states = np.repeat(node_size, len(G.nodes))
        colors = np.repeat(blue_color, len(G.nodes))

    for val, junction in zip(values_new, junctions_new):
        if val > 0: 
            G.add_edge(junction[0], junction[1], width=val)

    widths = [G[u][v]['width'] * 3 for u, v in G.edges] 

    if pos is None:
        pos = nx.kamada_kawai_layout(G=G, weight='width' if position_by_currents else None, seed=42)

    nx.draw(G=G, pos=pos, ax=ax, width=widths, arrowsize=arrowsize, node_size=states,
            edge_cmap=plt.cm.Reds, node_color=colors, edge_vmin=edge_vmin, edge_vmax=edge_vmax)

    return fig, ax