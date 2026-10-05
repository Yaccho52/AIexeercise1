# 时间序列预测对比工具

**在线试用**：[https://aiexeercise1-xlupd5wxqe8njmaamyhemr.streamlit.app/](https://aiexeercise1-xlupd5wxqe8njmaamyhemr.streamlit.app/)

本 python 程序是一个基于 AI 开发的、方便计算时间序列分析中预测值以及其他数值的工具。

对一组等间隔的时间序列，用四种经典模型分别做预测，并给出统一的评价指标：
**SSE / MSE / RMSE / 残差标准误 SE / Theil U1 / Theil U2**。

## 支持的模型

| 模型 | 公式 | 说明 |
|---|---|---|
| 线性趋势 Linear | $\hat y_t = a + b\,t$ | 最小二乘拟合后沿趋势外推 |
| 移动平均 MA | $\hat y_t = \frac{1}{k}\sum_{i=1}^{k} y_{t-i}$ | 最近 k 期的平均值 |
| 一次指数平滑 SES | $S_t = \alpha Y_t + (1-\alpha) S_{t-1}$ | 从初始值 S0 起逐期递推，第 t 期预测值取 $S_{t-1}$ |
| 朴素法 Naive | $\hat y_t = y_{t-1}$ | 上一期实际值，同时也是 Theil U2 的基准 |

## 评估方式：留出法

把数据从「预测起始期」处切成两段：前段只用来**建模**，后段只用来**检验**。
所有指标都按留出段的「实际值 vs 预测值」计算，衡量的是真正的样本外预测能力。

## 两种用法

### 一、网页应用（在线可操作）

`app.py` 是一个 Streamlit 应用：左侧栏勾选启用哪些模型、调整移动平均阶数、
SES 的 α 和初始值、预测起始期数；主区域粘贴数据，表格和图表实时更新。

```bash
pip install -r requirements.txt
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`。

**在线访问**：[https://aiexeercise1-xlupd5wxqe8njmaamyhemr.streamlit.app/](https://aiexeercise1-xlupd5wxqe8njmaamyhemr.streamlit.app/)

打开就能用，把链接发给任何人都行，对方不需要安装任何东西。（部署步骤见文末）

### 二、Jupyter notebook（适合阅读和分享）

`notebooks/` 文件夹下每个模型一个 notebook，**每个都完全独立**——
数据、公式、计算、绘图代码全部写在里面，单独打开就能运行：

| Notebook | 内容 |
|---|---|
| `01_线性趋势模型.ipynb` | 最小二乘拟合、回归预测区间 |
| `02_移动平均模型.ipynb` | k 期移动平均 |
| `03_一次指数平滑模型.ipynb` | S0 ~ Sn 的完整递推过程 |
| `04_朴素法模型.ipynb` | 朴素法与其作为 Theil U2 基准的关系 |
| `05_四模型对比与Excel导出.ipynb` | 四模型指标对比、综合图、导出 Excel |

```bash
pip install -r requirements-notebooks.txt
jupyter notebook
```

打开任意一本即可，然后 **从上到下依次运行**
（或者用菜单 `Kernel → Restart & Run All` 一次跑完整个 notebook）。

> notebook 里的变量是跨单元格共享的，**跳着只运行中间某一格会因为缺变量而失败**
> ——比如指标汇总那一格用到的是模型计算那一格的结果。
> 每个依赖前面结果的单元格开头都有一句 `_need(...)` 检查，
> 遇到这种情况它会直接告诉你缺哪个变量、该怎么做，而不会只抛一句 NameError。

图表已随文件保存，在 GitHub 上直接可见。

## 文件说明

| 文件 | 作用 |
|---|---|
| `app.py` | Streamlit 网页应用（交互界面） |
| `ts_core.py` | 计算核心，网页应用调用它 |
| `notebooks/` | 五个独立的 Jupyter notebook，按模型分类 |
| `requirements.txt` | 网页应用的依赖 |
| `requirements-notebooks.txt` | notebook 的依赖 |

换数据的方式：网页版直接在输入框里粘贴；notebook 改第 2 节的 `DATA` 和
`FORECAST_START`（从第几期开始预测，1 基）。

## 部署到 Streamlit Community Cloud

1. 确认仓库是**公开**的（Settings → General → 最下方 Change visibility）
2. 把本地改动 `git push` 上去
3. 打开 <https://share.streamlit.io>，用 GitHub 账号登录并授权
4. 点 **Create app**，填：Repository `Yaccho52/AIexeercise1`、Branch `main`、
   Main file path `app.py`
5. 点 **Deploy**，等 1~3 分钟构建完成后就能访问了

免费版**闲置一段时间会休眠**，下次访问需要十几秒唤醒；应用是公开的，
拿到链接的人都能用。之后每次 `git push`，网站会自动重新部署。

## 指标速查

- **SSE / MSE / RMSE / 残差标准误 SE**：越小，说明预测误差越小。
- **Theil U1**：0~1 之间的不等系数，越接近 0 越好，可以跨模型直接比较。
- **Theil U2**：以「每期都用上一期实际值」的朴素法为基准，
  **小于 1 表示优于朴素法，等于 1 持平，大于 1 更差**。
