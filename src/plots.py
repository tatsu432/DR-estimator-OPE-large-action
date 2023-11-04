import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns

plt.style.use("ggplot")

# # TSR with q
registered_colors = {
    "IPS": "tab:red",
    "DR": "tab:blue",
    "DM": "tab:purple",
    "MIPS": "tab:gray",
    "MDR": "tab:olive",
}

legend = ["IPS", "DR", "DM", "MIPS", "MDR"]







num_estimators = len(legend)


palette = [registered_colors[est] for est in legend]

# 各凡例の線を指定するためのカスタム凡例パッチ（線）を作成
line_legend_elements = [
    Line2D([0], [0], color=registered_colors[est], linewidth=5, marker='o', markerfacecolor=registered_colors[est], markersize=10, label=est) for est in legend
]

def plot_line(
    result_df, 
    x, 
    xlabel, 
    flag_log_scale, 
    flag_share_y_scale = True
) -> None:
    # 3つのプロットを1つの画像にまとめる
    fig, axes = plt.subplots(1, 3, figsize=(27, 7), tight_layout=True, sharey=flag_share_y_scale)

    # MSEのプロット（左側）
    sns.lineplot(
        linewidth=5,
        marker="o",
        markersize=8,
        markers=True,
        x=x,
        y="se",
        hue="est",
        ax=axes[0],
        palette=palette,
        data=result_df.query("(est == 'IPS' or est == 'DR' or est == 'DM' or est == 'MIPS' or est == 'MDR')"),

    )
    if flag_log_scale == True: 
        # y軸をlogスケールに設定
        axes[0].set_yscale("log")
    axes[0].set_xlabel("", fontsize=25)
    # axes[0].set_xlabel(xlabel, fontsize=25)
    axes[0].set_ylabel("")  
    axes[0].set_title("MSE", fontsize=25)
    axes[0].legend().set_visible(False)  
    axes[0].tick_params(axis='both', which='major', labelsize=20)  

    # Biasのプロット（中央）
    sns.lineplot(
        linewidth=5,
        marker="o",
        markersize=8,
        markers=True,
        x=x,
        y="bias",
        hue="est",
        ax=axes[1],
        palette=palette,
        data=result_df.query("(est == 'IPS' or est == 'DR' or est == 'DM' or est == 'MIPS' or est == 'MDR')"),
    )
    if flag_log_scale == True:
        # y軸をlogスケールに設定
        axes[1].set_yscale("log")
    axes[1].set_xlabel(xlabel, fontsize=25)
    axes[1].set_ylabel("")  
    axes[1].set_title("Bias", fontsize=25)
    axes[1].legend().set_visible(False)  
    axes[1].tick_params(axis='both', which='major', labelsize=20)  

    # Varianceのプロット（右側）
    sns.lineplot(
        linewidth=5,
        marker="o",
        markersize=8,
        markers=True,
        x=x,
        y="variance",
        hue="est",
        ax=axes[2],
        palette=palette,
        data=result_df.query("(est == 'IPS' or est == 'DR' or est == 'DM' or est == 'MIPS' or est == 'MDR')"),
    )
    if flag_log_scale == True:
        # y軸をlogスケールに設定
        axes[2].set_yscale("log")
    axes[2].set_xlabel("", fontsize=25)
    # axes[2].set_xlabel(xlabel, fontsize=25)
    axes[2].set_ylabel("")  
    axes[2].set_title("Variance", fontsize=25)
    axes[2].legend().set_visible(False)  
    axes[2].tick_params(axis='both', which='major', labelsize=20)  

    # 凡例を中央のプロットの真上に配置する
    fig.legend(handles=line_legend_elements, 
            loc='upper center', 
            bbox_to_anchor=(0.5, 1.15), 
            ncol=num_estimators, 
            fontsize=25)

    # 出力
    plt.show()