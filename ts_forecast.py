# -*- coding: utf-8 -*-
"""
==========================================================================
 ts_forecast.py —— 时间序列预测对比工具（留出法 / 生成 Excel 报告）
==========================================================================
把整段序列从"预测起始期"处切成两段：

    训练段（前若干期）—— 只用来建立模型
    留出段（后若干期）—— 只用来检验模型，不参与建模

模型在训练段上估计，然后预测留出段各期：移动平均一次性向后外推，
SES 从起始平滑值 S(T−1) 起逐期递推，Naive 取上一期实际值滚动预测（其 U2 恒为 1）。
报告中所有评价指标（SSE / MSE / RMSE / 标准误 SE / Theil U1、U2）都用
留出段的「实际值 vs 预测值」计算，衡量的是真正的样本外预测能力。

可选模型（显示顺序即报表顺序）
------------------------------
    Linear: 线性趋势模型，最小二乘拟合 y_t = a + b·t 后沿趋势外推
    MA    : k 期移动平均，外推值 = 训练段最后 k 期的平均值
    SES   : 一次指数平滑，从初始值 S(T−1) 起按 S_(t+1) = α·Y_t + (1−α)·S_t 递推
    Naive : 朴素法，第 T 期预测值 = 第 T-1 期实际值（滚动一步向前）

计算全部由 ts_core.py 完成（与网页版 app.py 共用同一套口径），
本文件只负责读取配置、打印摘要和写 Excel。

用法
----
    ① 在下面【用户配置区】填写 DATA、FORECAST_START、MA_ORDER 等
    ② 运行：python ts_forecast.py
    ③ 结果写入 ts_forecast_result.xlsx（脚本同目录）

依赖：openpyxl（仅此一个第三方库）；Python 3.8 及以上。
想用可视界面，见同目录的 app.py（streamlit run app.py）。
==========================================================================
"""

from __future__ import annotations

import math
import os

from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import ts_core as core


# ==========================================================================
# ①  用户配置区 —— 只需要修改这一段
# ==========================================================================

# 手动输入你的时间序列数据（按时间先后顺序排列）。
# 【示例数据】24 期销售额，请整段替换为你自己的数据。
DATA = [
    112, 118, 109, 121, 125, 117, 130, 128, 137, 134,
    141, 146, 139, 152, 148, 157, 161, 154, 166, 163,
    172, 175, 169, 181,
]

# 可选：与 DATA 等长的时点标签，如 ["2024-01", "2024-02", ...]。
# 填 None 则自动使用 T1, T2, T3 ...
TIME_LABELS = None

# 从第几期开始预测（1 基）。
#   填 6    →  用前 5 期建模，预测第 6 期及其后的全部观测值
#   填 None →  自动留出最后 HOLDOUT_TAIL 期用来检验
FORECAST_START = None
HOLDOUT_TAIL = 5          # FORECAST_START 为 None 时生效：留出最后几期

# 启用哪些模型：core.MODELS 的子集，可选 "Linear" / "MA" / "SES" / "Naive"
# 列表顺序即报表中的显示顺序
MODELS = ("Linear", "MA", "SES", "Naive")

# 移动平均的阶数 k（3 表示三期移动平均）
MA_ORDER = 3

# SES 的平滑系数 α，取值范围 0 < α < 1（越大越重视最新数据）
SES_ALPHA = 0.3

# SES 的初始值 S(T−1)（即第 T−1 期、训练段最后一期的预测值）：
#   填一个数值 → 直接采用；
#   填 None    → 自动取训练段实际值的算术平均（例如前 5 期的平均）
# 递推式 S_(t+1) = α·Y_t + (1−α)·S_t：第 T 期预测值 =
# α×(第 T−1 期实际值) + (1−α)×S(T−1)，之后每期用当期实际值和当期预测值算下期。
SES_LEVEL = None

# 预测区间的置信水平
CONFIDENCE = 0.95

# 输出文件名；None 表示保存为脚本同目录下的 ts_forecast_result.xlsx
OUTPUT_FILE = None

# 各模型的公式（写入 Excel 的「模型公式」工作表）
FORMULA_TEXT = {
    "MA": f"y_hat_t = (1/k)·Σ(i=1..k) y_(t-i)    其中 k = {MA_ORDER}",
    "SES": "S_(t+1) = α·Y_t + (1−α)·S_t    （S_t 为第 t 期预测值）",
    "Naive": "y_hat_t = y_(t−1)",
    "Linear": "y_hat_t = a + b·t；"
              "  b = Σ(t−t̄)(y_t−ȳ) / Σ(t−t̄)²；  a = ȳ − b·t̄",
}
FORMULA_NOTE = {
    "MA": "把最近 k 期的平均值作为预测值",
    "SES": "S(T−1) 为第 T−1 期的预测值（手动指定或取训练段实际值的平均值）：第 T 期预测值 = α×(第 T−1 期实际值)+(1−α)×S(T−1)，之后逐期递推",
    "Naive": "第 t 期的预测值取第 t−1 期的实际值（滚动一步向前）",
    "Linear": "对训练段 n 期做最小二乘拟合得到 a、b，再把期数 t 代入方程外推；"
              "预测标准误按回归预测区间公式计算，随外推步长增大",
}

# ==========================================================================
#   以下为实现部分，一般不需要修改
# ==========================================================================

_HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
_HEADER_FONT = Font(bold=True, color="FFFFFF", size=10)
_TITLE_FONT = Font(bold=True, size=13, color="1F4E79")
_NOTE_FONT = Font(italic=True, size=9, color="7F7F7F")
_BOLD = Font(bold=True)
_BEST_FILL = PatternFill("solid", fgColor="D9EAD3")
_SIDE = Side(style="thin", color="BFBFBF")
_BORDER = Border(left=_SIDE, right=_SIDE, top=_SIDE, bottom=_SIDE)
_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
_NUM_FMT = "0.0000"


def _write_title(ws, row, text, span):
    c = ws.cell(row=row, column=1, value=text)
    c.font = _TITLE_FONT
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=span)
    ws.row_dimensions[row].height = 20


def _write_note(ws, row, text, span):
    c = ws.cell(row=row, column=1, value=text)
    c.font = _NOTE_FONT
    c.alignment = Alignment(vertical="center", wrap_text=False)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=span)
    ws.row_dimensions[row].height = 16


def _write_table(ws, start_row, headers, rows, num_fmt=_NUM_FMT, start_col=1):
    """写入带表头样式的表格，返回最后一行行号。"""
    for j, h in enumerate(headers):
        c = ws.cell(row=start_row, column=start_col + j, value=h)
        c.fill = _HEADER_FILL
        c.font = _HEADER_FONT
        c.alignment = _CENTER
        c.border = _BORDER
    ws.row_dimensions[start_row].height = 30

    r = start_row
    for i, row_vals in enumerate(rows):
        r = start_row + 1 + i
        for j, v in enumerate(row_vals):
            c = ws.cell(row=r, column=start_col + j, value=v)
            c.border = _BORDER
            c.alignment = Alignment(horizontal="center", vertical="center")
            if isinstance(v, float):
                c.number_format = num_fmt
    return r


def _set_widths(ws, widths):
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def _highlight_best(ws, first_row, last_row, col):
    """把某列中的最小值标记为浅绿色（该列指标越小越好）。"""
    vals = []
    for r in range(first_row, last_row + 1):
        v = ws.cell(row=r, column=col).value
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            vals.append((v, r))
    if not vals:
        return
    best = min(vals)[1]
    ws.cell(row=best, column=col).fill = _BEST_FILL
    ws.cell(row=best, column=col).font = _BOLD


def build_workbook(y, results, info, labels, z, wb):
    """把结果写成 Excel 工作簿。

    工作表：输入数据 / 模型公式 / 训练段拟合 / 留出段预测 / 预测区间 / 模型对比 / 走势图
    """
    n, train_n, horizon = info["n"], info["train_n"], info["horizon"]
    t_index = info["t_index"]

    # ===== Sheet 1 : 输入数据 =====
    ws = wb.active
    ws.title = "输入数据"
    _write_title(ws, 1, "输入数据", 4)
    _write_note(ws, 2, f"共 {n} 期观测值：前 {train_n} 期用于建模，"
                       f"第 {info['forecast_start']} ~ {n} 期用于检验", 4)
    _write_table(ws, 4, ["序号", "时点", "观测值 y_t", "用途"],
                 [[i + 1, labels[i], y[i],
                   "训练段（建模）" if i < train_n else "留出段（检验）"]
                  for i in range(n)], num_fmt="0.0000")
    _set_widths(ws, [8, 14, 14, 16])
    ws.freeze_panes = "A5"

    # ===== Sheet 2 : 模型公式 =====
    wsf = wb.create_sheet("模型公式")
    _write_title(wsf, 1, "各模型的数学公式", 4)
    _write_note(wsf, 2, "记号：t 为期数（1, 2, 3…），y_t 为第 t 期实际值，"
                        "y_hat_t 为第 t 期预测值，n 为训练段期数，T 为训练段末期数", 4)
    frows = []
    for res in results:
        frows.append([res.name, res.params,
                      FORMULA_TEXT[res.id], FORMULA_NOTE[res.id]])
        if res.id == "Linear":
            fi = res.fit_info
            frows.append(["（拟合结果）", "",
                          f"y_hat_t = {fi['a']:.4f} + {fi['b']:.4f}·t",
                          f"s = {fi['s']:.4f}，n = {fi['n']}，t̄ = {fi['xbar']:.4f}，"
                          f"Σ(t−t̄)² = {fi['sxx']:.4f}"])
    lastf = _write_table(wsf, 4, ["模型", "参数", "公式", "说明"], frows)
    for rr in range(5, lastf + 1):
        for col in (1, 2, 3, 4):
            wsf.cell(row=rr, column=col).alignment = Alignment(
                horizontal="left", vertical="center", wrap_text=True)
    _set_widths(wsf, [20, 22, 44, 44])

    # ===== Sheet 3 : 训练段拟合 =====
    ws2 = wb.create_sheet("训练段拟合")
    _write_title(ws2, 1, "训练段：一步向前拟合与误差", 2 + 2 * len(results))
    _write_note(ws2, 2, "仅用于观察各模型在建模数据上的贴合程度；"
                        "空白表示该模型在该时点尚无法给出拟合值", 2 + 2 * len(results))
    headers = ["时点", "观测值 y_t"]
    for res in results:
        headers += [f"{res.key} 拟合值", f"{res.key} 误差"]
    aligned = [(r.aligned_train_fitted(n), r.aligned_train_errors(n))
               for r in results]
    rows = []
    for i in range(train_n):
        row = [labels[i], y[i]]
        for fit_a, err_a in aligned:
            row += [fit_a[i], err_a[i]]
        rows.append(row)
    last = _write_table(ws2, 4, headers, rows, num_fmt="0.0000")
    _set_widths(ws2, [12, 12] + [14] * (2 * len(results)))
    ws2.freeze_panes = "C5"

    ch = LineChart()
    ch.title = "训练段：观测值与各模型拟合值"
    ch.style = 2
    ch.y_axis.title = "观测值"
    ch.x_axis.title = "时点"
    data = Reference(ws2, min_col=2, max_col=2 + 2 * len(results),
                     min_row=4, max_row=last)
    cats = Reference(ws2, min_col=1, min_row=5, max_row=last)
    ch.add_data(data, titles_from_data=True)
    ch.set_categories(cats)
    ch.height, ch.width = 9, 22
    ch.dispBlanksAs = "gap"
    ch.y_axis.numFmt = "0"
    ch.x_axis.delete = False
    ch.y_axis.delete = False
    ws2.add_chart(ch, f"A{last + 3}")

    # ===== Sheet 4 : 留出段预测 =====
    ws3 = wb.create_sheet("留出段预测")
    _write_title(ws3, 1, "留出段：各模型预测值与实际值对照", 2 + 2 * len(results))
    _write_note(ws3, 2, "误差 = 实际值 − 预测值；Naive 的预测值取上一期实际值，本表就是评价依据",
                2 + 2 * len(results))
    headers = ["时点", "实际值"]
    for res in results:
        headers += [f"{res.key} 预测值", f"{res.key} 误差"]
    rows = []
    for i, t in enumerate(t_index):
        row = [labels[t], y[t]]
        for res in results:
            row += [res.forecast[i], res.errors[i]]
        rows.append(row)
    _write_table(ws3, 4, headers, rows, num_fmt="0.0000")
    _set_widths(ws3, [12, 12] + [14] * (2 * len(results)))
    ws3.freeze_panes = "C5"

    # ===== Sheet 5 : 预测标准误与区间 =====
    ws4 = wb.create_sheet("预测区间")
    _write_title(ws4, 1, f"留出段预测标准误与 {CONFIDENCE:.0%} 预测区间",
                 1 + 3 * len(results))
    _write_note(ws4, 2, "SE(h) 为第 h 期预测标准误；预测区间 = 预测值 ± z·SE(h)，"
                        f"z = {z:.4f}", 1 + 3 * len(results))
    headers = ["时点"]
    for res in results:
        headers += [f"{res.key} SE(h)", f"{res.key} 下限", f"{res.key} 上限"]
    rows = []
    for i, t in enumerate(t_index):
        row = [labels[t]]
        for res in results:
            fc, se = res.forecast[i], res.se_pred[i]
            row += [se, fc - z * se, fc + z * se]
        rows.append(row)
    _write_table(ws4, 4, headers, rows, num_fmt="0.0000")
    _set_widths(ws4, [12] + [14] * (3 * len(results)))
    ws4.freeze_panes = "B5"

    # ===== Sheet 6 : 模型对比 =====
    ws5 = wb.create_sheet("模型对比")
    _write_title(ws5, 1, "模型对比（留出段）", 10)
    _write_note(ws5, 2, f"训练段 = 第 1 ~ {train_n} 期；"
                        f"留出段 = 第 {info['forecast_start']} ~ {n} 期"
                        f"（H = {horizon}）；所有指标都在留出段上计算。", 10)
    cmp_headers = ["模型", "参数", "留出期数 H", "SSE", "MSE", "RMSE",
                   "自由度 df", "残差标准误 SE", "Theil U1", "Theil U2"]
    cmp_rows = []
    for res in results:
        s = res.stats
        cmp_rows.append([res.name, res.params, s["n"], s["sse"], s["mse"],
                         s["rmse"], s["df"], s["se"], s["u1"], s["u2"]])
    last5 = _write_table(ws5, 4, cmp_headers, cmp_rows, num_fmt="0.0000")
    for r in range(5, last5 + 1):
        ws5.cell(row=r, column=1).alignment = Alignment(
            horizontal="left", vertical="center")
        ws5.cell(row=r, column=2).alignment = Alignment(
            horizontal="left", vertical="center")
    _set_widths(ws5, [20, 14, 12, 12, 12, 11, 10, 14, 11, 11])
    for col in (4, 5, 6, 8, 9, 10):   # SSE, MSE, RMSE, SE, U1, U2 —— 越小越好
        _highlight_best(ws5, 5, last5, col)
    note_row = last5 + 2
    for i, txt in enumerate([
        "判读说明：",
        "  · 指标全部按留出段的「实际值 − 预测值」计算，反映样本外预测能力。",
        "  · SSE / MSE / RMSE / SE 越小越好；残差标准误 SE = √(SSE/(H−k))，"
        "k 为待估参数个数（SES 计 1 个，其余为 0）。",
        "  · Theil U1 ∈ [0,1]，越接近 0 越好，可跨模型直接比较。",
        "  · Theil U2 以「每期都用上一期实际值」的朴素预测为基准："
        "本表中 Naive 即为该基准，其 U2 = 1；其余模型 U2 < 1 才算优于朴素法。",
    ]):
        c = ws5.cell(row=note_row + i, column=1, value=txt)
        c.font = _BOLD if i == 0 else _NOTE_FONT

    # ===== Sheet 7 : 走势图 =====
    ws6 = wb.create_sheet("走势图")
    _write_title(ws6, 1, "观测值与留出段预测", 2 + len(results))
    tail = min(8, n)
    plot_rows = []
    for i in range(n - tail, n):          # 最近 tail 期的观测值
        plot_rows.append([labels[i], y[i]] + [None] * len(results))
    # 连接点：训练段最后一期补上各模型的起点，使折线连续
    last_hist = None
    for idx, row in enumerate(plot_rows):
        if row[0] == labels[train_n - 1]:
            last_hist = idx
            break
    if last_hist is not None:
        for j in range(len(results)):
            plot_rows[last_hist][2 + j] = y[train_n - 1]
    for i, t in enumerate(t_index):       # 留出段各期预测值
        plot_rows.append([labels[t], None] + [r.forecast[i] for r in results])

    _write_table(ws6, 3, ["时点", "观测值"] + [f"{r.key} 预测" for r in results],
                 plot_rows, num_fmt="0.0000")
    end6 = 3 + len(plot_rows)
    _set_widths(ws6, [14, 12] + [13] * len(results))

    ch6 = LineChart()
    ch6.title = "观测值与各模型留出段预测"
    ch6.style = 2
    ch6.y_axis.title = "观测值"
    ch6.x_axis.title = "时点"
    d6 = Reference(ws6, min_col=2, max_col=2 + len(results), min_row=3, max_row=end6)
    c6 = Reference(ws6, min_col=1, min_row=4, max_row=end6)
    ch6.add_data(d6, titles_from_data=True)
    ch6.set_categories(c6)
    ch6.height, ch6.width = 10, 24
    ch6.dispBlanksAs = "gap"
    ch6.y_axis.numFmt = "0"
    ch6.x_axis.delete = False
    ch6.y_axis.delete = False
    ws6.add_chart(ch6, "H3")
    return wb


def main():
    y = [float(v) for v in DATA]
    n = len(y)
    if n < 4:
        raise ValueError("至少需要 4 期观测值（前段建模 + 后段检验）。")

    labels = list(TIME_LABELS) if TIME_LABELS else [f"T{i + 1}" for i in range(n)]
    if len(labels) != n:
        raise ValueError("TIME_LABELS 长度必须与 DATA 一致。")

    forecast_start = FORECAST_START
    if forecast_start is None:
        forecast_start = max(3, n - HOLDOUT_TAIL + 1)
    forecast_start = int(forecast_start)
    if not 3 <= forecast_start <= n - 1:
        raise ValueError(
            f"FORECAST_START 需要介于 3 和 {n - 1} 之间（留出段至少 1 期）。")

    results, info = core.run_holdout(
        y, forecast_start, models=MODELS, ma_order=MA_ORDER,
        ses_level=SES_LEVEL, alpha=SES_ALPHA, confidence=CONFIDENCE)
    z = core.z_value(CONFIDENCE)
    train_n, horizon = info["train_n"], info["horizon"]

    # ---- 控制台摘要 ----
    print("=" * 84)
    print(f"时间序列预测对比（留出法）   总期数 n = {n}   "
          f"移动平均阶数 = {MA_ORDER}")
    print(f"训练段：第 1 ~ {train_n} 期（建模）　"
          f"留出段：第 {info['forecast_start']} ~ {n} 期（检验，H = {horizon}）")
    for _r in results:
        if _r.id == "SES":
            _fi = _r.fit_info
            print(f"SES：α = {_fi['alpha']:g}，初始值 S(T−1) = {_fi['level']:.4f}"
                  f"（{'手动指定' if _fi['given'] else '训练段实际值平均'}）")
    print("=" * 84)
    print(f"{'模型':<22}{'SSE':>12}{'RMSE':>10}{'SE':>10}{'U1':>9}{'U2':>9}")
    print("-" * 84)
    for res in results:
        s = res.stats
        print(f"{res.name:<22}{s['sse']:>12.4f}{s['rmse']:>10.4f}"
              f"{s['se']:>10.4f}{s['u1']:>9.4f}{s['u2']:>9.4f}")
    print("-" * 84)
    print("留出段：实际值 vs 各模型预测")
    print(f"{'时点':<8}{'实际值':>12}" + "".join(f"{r.key:>12}" for r in results))
    for i, t in enumerate(info["t_index"]):
        print(f"{labels[t]:<8}{y[t]:>12.4f}"
              + "".join(f"{r.forecast[i]:>12.4f}" for r in results))
    print("=" * 84)

    # ---- 写 Excel ----
    wb = build_workbook(y, results, info, labels, z, Workbook())
    out = OUTPUT_FILE or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ts_forecast_result.xlsx")
    wb.save(out)
    print(f"\n结果已保存：{out}")
    return out


if __name__ == "__main__":
    main()
