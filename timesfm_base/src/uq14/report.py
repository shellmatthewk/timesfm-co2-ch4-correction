"""Read saved results only; create ordered Chinese summaries and static figures."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from .workflow import ROOT, write_json

LABELS = {"native": "原始TimesFM", "deterministic": "MSE模型（3种子均值）", "engression": "Engression（3种子均值）"}


def group_of(name):
    return "native" if name == "native_timesfm" else name.split("_seed")[0]


def averaged(dicts):
    keys = set().union(*(d.keys() for d in dicts))
    return {k: float(np.mean([d[k] for d in dicts])) for k in keys
            if all(isinstance(d.get(k), (int, float)) and not isinstance(d.get(k), bool) for d in dicts)}


def fmt(value, places=4, percent=False):
    if value is None:
        return "不适用"
    return f"{100*value:.1f}%" if percent else f"{value:.{places}f}"


def save_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def report():
    source = json.loads((ROOT / "07_评估结果/all_results.json").read_text())
    groups = {}
    for group in LABELS:
        runs = [r for r in source["models"] if group_of(r["model"]) == group]
        groups[group] = {"label": LABELS[group], "runs": [r["model"] for r in runs],
            "before": averaged([r["before"]["overall"] for r in runs]),
            "after": averaged([r["after_conformal"]["overall"] for r in runs]),
            "by_station": {s: {phase: averaged([r[path]["by_station"][s] for r in runs])
                                for phase,path in (("before","before"),("after","after_conformal"))}
                           for s in ("BRW","MLO","SMO","SPO")},
            "by_lead": {h: {phase: averaged([r[path]["by_lead"][h] for r in runs])
                             for phase,path in (("before","before"),("after","after_conformal"))}
                        for h in map(str,range(1,15))}}
        for phase in ("before","after"):
            if "coverage80" in groups[group][phase]:
                groups[group][phase]["coverage_gap_percentage_points"] = 100*abs(groups[group][phase]["coverage80"]-.8)
    write_json(ROOT / "07_评估结果/summary.json", {"groups": groups,
        "aggregation": "Each seed scored separately; then arithmetic mean. Not an ensemble.",
        "accuracy_definition": "MAE and RMSE in ppm; no invented classification accuracy percentage.",
        "joint_full_coverage": None, "complete_test_windows": 0, "observed_test_cells": 676})
    rows = []
    for group,g in groups.items():
        for phase in ("before","after"):
            a = g[phase]
            rows.append({"method": group,"phase":phase,"mae_ppm":a["mae"],"rmse_ppm":a["rmse"],
                "coverage80":a.get("coverage80"),"coverage_gap_percentage_points":a.get("coverage_gap_percentage_points"),
                "width80_ppm":a.get("mean_width80"),"mis80_ppm":a.get("mis80"),
                "wis9_ppm":a.get("wis9"),"crps_ppm":a.get("crps_unbiased")})
    save_csv(ROOT / "07_评估结果/summary.csv", rows)
    for group_name, key in (("station","by_station"),("lead","by_lead")):
        rows=[]
        for model in source["models"]:
            for phase in ("before","after_conformal"):
                for label,m in model[phase][key].items():
                    rows.append({"model":model["model"],"phase":phase,group_name:label,"n":m["n"],
                        "mae_ppm":m["mae"],"rmse_ppm":m["rmse"],"coverage80":m.get("coverage80"),
                        "width80_ppm":m.get("mean_width80"),"mis80_ppm":m.get("mis80"),
                        "wis9_ppm":m.get("wis9"),"crps_ppm":m.get("crps_unbiased")})
        save_csv(ROOT / "07_评估结果" / f"by_{group_name}.csv",rows)
    rows=[]
    for r in source["models"]:
        p=ROOT/'06_共形校准'/r['model']/'correction.npz'
        with np.load(p) as a:
            for i,s in enumerate(("BRW","MLO","SMO","SPO")):
                for h in range(14):
                    rows.append({"model":r['model'],"station":s,"lead":h+1,
                                 "cal_n":int(a['cal_n'][i,h]),"rank":int(a['rank'][i,h]),
                                 "raw_q_ppm":float(a['raw_q'][i,h]),"expansion_ppm":float(a['q'][i,h])})
    save_csv(ROOT/'06_共形校准/calibration_parameters.csv',rows)
    lines=["# 07 结果：准确性与共形校准前后对照", "",
        "本报告读取已保存的固定模型预测。预测任务为四站各自过去14天→未来14天，真实目标是站点日值。2025年是已在其他任务使用过的回溯开发期。",
        "", "## 1. 点预测准确性", "",
        "accuracy采用连续值回归指标MAE、RMSE，单位ppm，越小越好。没有将1−MAE或覆盖率称为准确率。Engression取512条轨迹的逐点中位数。",
        "", "|方法|MAE（ppm）|RMSE（ppm）|校准是否改变点预测|", "|---|---:|---:|---|"]
    for g in groups.values():
        lines.append(f"|{g['label']}|{fmt(g['before']['mae'])}|{fmt(g['before']['rmse'])}|否|")
    lines += ["", "## 2. 校准质量（名义覆盖率80%）", "",
        "关注实测覆盖率与80%的差距，同时比较宽度和平均区间评分（MIS）。覆盖率增加本身不等于整体概率预测质量提高。每个站点、预测步独立校准，扩张量非负。",
        "", "|方法|覆盖率：前→后|与80%差距：前→后（百分点）|平均宽度：前→后（ppm）|MIS：前→后（ppm）|",
        "|---|---:|---:|---:|---:|"]
    for g in groups.values():
        b,a=g['before'],g['after']
        pair=lambda k,p=False: f"{fmt(b.get(k),percent=p)} → {fmt(a.get(k),percent=p)}"
        lines.append(f"|{g['label']}|{pair('coverage80',True)}|{pair('coverage_gap_percentage_points')}|{pair('mean_width80')}|{pair('mis80')}|")
    lines += ["", "MSE模型校准前仅有点预测，未定义概率区间。其校准后区间来自独立校准期的绝对残差。原始TimesFM与Engression使用Q10/Q90做扩张式CQR。",
        "", "## 3. 原始分布评分", "",
        "|方法|九分位加权区间评分WIS|连续秩概率评分CRPS|观测子向量归一化能量评分|", "|---|---:|---:|---:|"]
    for g in groups.values():
        b=g['before'];lines.append(f"|{g['label']}|{fmt(b.get('wis9'))}|{fmt(b.get('crps_unbiased'))}|{fmt(b.get('energy_score_observed_normalized'))}|")
    lines += ["", "原生TimesFM只有9个分位点，无法唯一确定完整CRPS。共形后只修改80%区间边界，没有构造新的完整分布，故不报告虚构的校准后CRPS或WIS9；可比较校准前后的MIS80。",
        "", "## 4. 四个站点分别的结果", "",
        "|站点|方法|MAE（ppm）|RMSE（ppm）|覆盖率：前→后|宽度：前→后（ppm）|", "|---|---|---:|---:|---:|---:|"]
    for station in ("BRW","MLO","SMO","SPO"):
        for g in groups.values():
            b,a=g['by_station'][station]['before'],g['by_station'][station]['after']
            lines.append(f"|{station}|{g['label']}|{fmt(b['mae'])}|{fmt(b['rmse'])}|{fmt(b.get('coverage80'),percent=True)} → {fmt(a.get('coverage80'),percent=True)}|{fmt(b.get('mean_width80'))} → {fmt(a.get('mean_width80'))}|")
    lines += ["", "## 5. 训练选择与解释范围", "",
        "|模型|验证集选中的轮数|训练至停止轮数|", "|---|---:|---:|"]
    for r in source['models']:
        if r['model']=='native_timesfm':continue
        d=json.loads((ROOT/'05_训练'/r['model']/'complete.json').read_text())
        lines.append(f"|{r['model']}|{d['best_epoch']}|{d['epochs_run']}|")
    lines += ["", "三种子均值是分别计分再平均，不是合并样本形成集成。所有预先指定运行均保留。",
        "", "测试仅15个起点、676/840个真实站点日值，未来缺失值完全不参与计分。验证、校准和测试期均没有完整56维标签窗口，因此无法实测完整四站两周的同时覆盖率。数据点之间相关，不能将676个观测解释为676个独立样本。",
        "", "校准仅13个起点，每个站点×预测步只有7–13个残差。时间依赖、季节变化和缺测选择可能破坏交换性；这里的共形结果是实际回溯覆盖率，不是已建立的分布无关80%保证。预训练数据重叠没有排除，未运行A100。",
        "", "## 6. 对应文件", "",
        "- `summary.csv`：三类模型汇总。", "- `all_results.json`：全部7个模型的未取整指标。",
        "- `by_station.csv`、`by_lead.csv`：逐站、逐预测步的校准前后结果。",
        "- `predictions_all_methods.csv`：预测、真实值、缺失标记与区间。",
        "- 各模型子目录的 `test_predictions.npz`：包括Engression512条样本，可独立重新评分。",
        "- `../06_共形校准/calibration_parameters.csv`：全部56个校准单元的样本量、秩与阈值。", ""]
    (ROOT/'07_评估结果/07_结果与校准对照.md').write_text('\n'.join(lines))
    figures(groups)
    print("Saved Chinese result report, CSVs, summary.json and figures.",flush=True)


def figures(groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available={f.name for f in font_manager.fontManager.ttflist}
    if 'Arial Unicode MS' in available: plt.rcParams['font.family']='Arial Unicode MS'
    plt.rcParams['axes.unicode_minus']=False
    out=ROOT/'07_评估结果/figures';out.mkdir(exist_ok=True)
    colors={'native':'#6a7f91','deterministic':'#c48633','engression':'#007f86'}
    fig,axs=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    names=list(groups)
    for j,metric in enumerate(('coverage80','mean_width80')):
        ax=axs[j]
        for i,name in enumerate(names):
            g=groups[name];b=g['before'].get(metric);a=g['after'][metric]
            if b is not None: ax.bar(i-.17,b*(100 if j==0 else 1),width=.30,color=colors[name],alpha=.4)
            ax.bar(i+.17,a*(100 if j==0 else 1),width=.30,color=colors[name])
        ax.set_xticks(range(3),['原始TimesFM','MSE模型','Engression'])
        ax.set_ylabel('实测覆盖率（%）' if j==0 else '平均区间宽度（ppm）')
        ax.spines[['top','right']].set_visible(False)
        if j==0:ax.axhline(80,color='#333333',linestyle='--',label='目标80%');ax.legend();ax.set_ylim(0,105)
    fig.suptitle('浅色：校准前  /  深色：校准后；MSE校准前没有概率区间',fontsize=13)
    fig.savefig(out/'conformal_before_after.png',dpi=180);fig.savefig(out/'conformal_before_after.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    x=np.arange(4)
    for i,(name,g) in enumerate(groups.items()):
        vals=[g['by_station'][s]['before']['mae'] for s in ('BRW','MLO','SMO','SPO')]
        ax.bar(x+(i-1)*.24,vals,width=.23,label=g['label'],color=colors[name])
    ax.set_xticks(x,['BRW','MLO','SMO','SPO']);ax.set_ylabel('平均绝对误差（ppm，越小越好）');ax.legend()
    ax.spines[['top','right']].set_visible(False)
    ax.set_title('四站点预测准确性；共形校准不改变点预测')
    fig.savefig(out/'accuracy_by_station.png',dpi=180);fig.savefig(out/'accuracy_by_station.svg');plt.close(fig)


if __name__=='__main__': report()
