from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
def visualize(ego_in, agents_in, agents_in_mask, roads, roads_mask, radius=80, ruler=True, save=True,
              return_PIL_image=False, legend=True, save_addr="viz/1.png"):
    """
    Visualizes the ego vehicle, other agents, and roads.

    Args:
        ego_in (numpy.ndarray): Array of ego vehicle coordinates.
        agents_in (numpy.ndarray): Array of other agents' coordinates.
        roads (list): List of road coordinates.
        radius (int, optional): Radius of the visualization area. Defaults to 80.
        ruler (bool, optional): Whether to display a ruler. Defaults to True.
        save (bool, optional): Whether to save the visualization. Defaults to True.
        return_PIL_image (bool, optional): Whether to return the visualization as a PIL image. Defaults to False.
        legend (bool, optional): Whether to display a legend. Defaults to True.
        save_addr (str, optional): Address to save the visualization. Defaults to "viz/1.png".

    Returns:
        PIL.Image.Image or None: The visualization as a PIL image if return_PIL_image is True, otherwise None.
    """
    test=0
    fig = plt.figure(dpi=300)
    x_min, x_max = ego_in[-1, 0] - radius, ego_in[-1, 0] + radius
    y_min, y_max = ego_in[-1, 1] - radius, ego_in[-1, 1] + radius
    plt.xlim(x_min, x_max)
    plt.ylim(y_min,y_max)
    plt.axis('off')
    arrow_scale_factor = radius / 50

    color_dict = {"AGENT_HISTORY": "#a6961b", "AGENT_GT": "#006B73",
                  "OTHERS": "#929eea", "AV": "#007672", "AGENT_HISTORY_ORIG": "#804600",
                  "AGENT_PRED_ORIG": "#DE00CB", "Road_color": "#A5A5A3",
                  "AGENT_PREDS": ["#fef001", "#ffce03", "#fd9a01", "#fd6104", "#ff2c05"]}
    
    # plot roads
    
    for road, mask in zip(roads, roads_mask):  
     
        road = road[mask[:] == 1, :3]
        #if len(road) == 0:
        #    continue
        ###if(i==closest_idx):
        ##    plt.plot(road[:, 0], road[:, 1], color="red", linewidth=1)
        #else:
        plt.plot(road[:, 0], road[:, 1], color=color_dict["Road_color"], linewidth=1)

    # plot ego
    ego_line = plt.plot(
        ego_in[:, 0],
        ego_in[:, 1],
        "-",
        color=color_dict["AGENT_HISTORY"],
        alpha=1,
        linewidth=2,
        zorder=1,
    )
    plt.scatter(ego_in[:, 0], ego_in[:, 1], color=color_dict["AGENT_HISTORY"], s=5)

    # plot other agents
    others_line = None
   
    for agent, mask in zip(agents_in, agents_in_mask):
        agent_history = agent[mask[:] == 1, :3]
        agent_traj = agent_history
        if agent_traj.shape[0] >= 3:
            others_line = plt.plot(
                agent_traj[:, 0],
                agent_traj[:, 1],
                "-",
                color=color_dict["OTHERS"],
                alpha=1,
                linewidth=1.5,
                zorder=0,
            )
            plt.scatter(agent_traj[:, 0], agent_traj[:, 1], color=color_dict["OTHERS"], s=5)
            #def plot_arrow(current_data, color, scale_factor, arrow_scale_factor):
            #    m, b = np.polyfit(current_data[-3:, 0], current_data[-3:, 1], 1)
            #    plt.arrow(current_data[-2, 0], current_data[-2, 1],
            #              (current_data[-1, 0] - current_data[-2, 0]) / np.abs(
            #                  current_data[-1, 0] - current_data[-2, 0]) / np.sqrt(1 + m ** 2),
            #              m * (current_data[-1, 0] - current_data[-2, 0]) / np.abs(
            #                  current_data[-1, 0] - current_data[-2, 0]) / np.sqrt(1 + m ** 2)
            #              , color=color, width=0.1 * arrow_scale_factor * scale_factor,
            #              head_width=0.02 * arrow_scale_factor * scale_factor * 8,
            #              head_length=0.02 * arrow_scale_factor * scale_factor * 7, zorder=0, length_includes_head=True)

            #plot_arrow(agent_traj, color_dict["OTHERS"], 1, arrow_scale_factor)
    

    if ruler:
        plt.plot(np.array([ego_in[-1, 0] - 0.9 * radius, ego_in[-1, 0] - 0.7 * radius]),
                 np.array([ego_in[-1, 1] - 0.9 * radius, ego_in[-1, 1] - 0.9 * radius]), color="black", linewidth=3)
        plt.plot(np.array([ego_in[-1, 0] - 0.91 * radius, ego_in[-1, 0] - 0.91 * radius]),
                 np.array([ego_in[-1, 1] - 0.88 * radius, ego_in[-1, 1] - 0.92 * radius]), color="black", linewidth=2)
        plt.plot(np.array([ego_in[-1, 0] - 0.7 * radius, ego_in[-1, 0] - 0.7 * radius]),
                 np.array([ego_in[-1, 1] - 0.88 * radius, ego_in[-1, 1] - 0.92 * radius]), color="black", linewidth=2)

        plt.text(ego_in[-1, 0] - 0.87 * radius, ego_in[-1, 1] - 0.85 * radius, str(int(radius * 0.2)) + "m")

    if legend:
        lines, names = [], []
        for line, name in zip([others_line, ego_line], ["Other Agents", "Ego Past"]):
            if line is not None:
                lines.append(line[0])
                names.append(name)
        plt.legend(handles=lines, labels=names, loc="upper right")

    if save:
        if not os.path.exists(save_addr[: save_addr.rfind('/')]):
            os.makedirs(save_addr[: save_addr.rfind('/')])
        plt.savefig(save_addr[: save_addr.rfind('/') + 1] + "PNG_" + save_addr[save_addr.rfind('/') + 1:] + ".png",
                    bbox_inches='tight')
        plt.savefig(save_addr[: save_addr.rfind('/') + 1] + "PDF_" + save_addr[save_addr.rfind('/') + 1:] + ".pdf",
                    bbox_inches='tight')
        if not return_PIL_image:
            plt.close()

    if return_PIL_image:
        fig.canvas.draw()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close()
        return img