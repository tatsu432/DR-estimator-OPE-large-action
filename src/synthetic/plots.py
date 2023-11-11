import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import conf

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

y_list = ["se", "bias", "variance"]
title_list = ["MSE", "Squared Bias", "Variance"]

def plot_line(
    result_df, 
    x, 
    xlabel, 
    xticklabels = None,
    flag_log_scale: bool = False, 
    flag_share_y_scale: bool = True
) -> None:
    # 3つのプロットを1つの画像にまとめる
    fig, axes = plt.subplots(1, 3, figsize=(27, 7), tight_layout=True, sharey=flag_share_y_scale)


    for i in range(3):
        # MSEのプロット（左側）
        sns.lineplot(
            linewidth=5,
            marker="o",
            markersize=conf.markersize,
            markers=True,
            x=x,
            y=y_list[i],
            hue="est",
            ax=axes[i],
            palette=palette,
            data=result_df.query("(est == 'IPS' or est == 'DR' or est == 'DM' or est == 'MIPS' or est == 'MDR')"),

        )
        # if x == "n_rounds" or "n_action": 
        #     # y軸をlogスケールに設定
        #     axes[i].set_xscale("log")
        if flag_log_scale == True: 
            # y軸をlogスケールに設定
            axes[i].set_yscale("log")
        if i == 1:
            axes[i].set_xlabel(xlabel, fontsize=25)
        else:
            axes[i].set_xlabel("", fontsize=25)
        if xticklabels != None:
            axes[i].set_xticks(xticklabels)
            axes[i].set_xticklabels(xticklabels, fontsize=18)
        axes[i].set_ylabel("")  
        axes[i].set_title(title_list[i], fontsize=25)
        axes[i].legend().set_visible(False)  
        axes[i].tick_params(axis='both', which='major', labelsize=20)  



    # 凡例を中央のプロットの真上に配置する
    fig.legend(handles=line_legend_elements, 
            loc='upper center', 
            bbox_to_anchor=(0.5, 1.15), 
            ncol=num_estimators, 
            fontsize=25)

    # 出力
    plt.show()