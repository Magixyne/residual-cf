# MovieLens 残差协同过滤

普通 Python 项目：双偏差基线、残差 UserCF/ItemCF、混合推荐、离线对照评估和终端 TUI。数据与预计算推荐缓存随仓库提供，启动演示不需要联网或重训。

## 项目结构

- main.py：启动 TUI、查看单用户推荐、运行评估或重建推荐缓存。
- recommender.py：数据读取、时序切分、模型、参数选择、评估与推荐缓存。
- tui.py：基于 prompt_toolkit 的终端交互界面。
- requirements.txt：运行依赖。
- data/ml-latest-small/：MovieLens CSV、预计算推荐缓存和评估结果。
- report/：LaTeX 源码、编译后的实验报告及图表资源。

说明：report/report.tex 已按当前源码与 data/ml-latest-small/evaluation_results.json 校正，report.pdf 由该源码重新编译；代码附录通过 listings 直接引用 recommender.py。

## 安装与运行

需要 Python 3.10 或更新版本。

    python -m pip install -r requirements.txt
    python main.py

查看用户 66 的推荐：

    python main.py --once 66

重新运行训练内调参与时间留出评估：

    python main.py --evaluate

交互界面支持用户 ID、流派过滤和片名搜索。按 F2/F3 翻页、Tab 切换输入框、Ctrl-Q 退出。若要重训全量模型并重建缓存，运行 python main.py --rebuild。

## 数据与结果

数据目录包含 ratings.csv、movies.csv、links.csv 和 tags.csv，对应 100,004 条评分、671 位用户、9,066 部有评分电影和 9,125 部目录电影。数据来源为 [GroupLens MovieLens](https://grouplens.org/datasets/movielens/)；再次分发时请遵循其使用条款。

当前随附评估结果为：偏差基线 RMSE/MAE 0.9072/0.6949，UserCF 0.9016/0.6887，ItemCF 0.9015/0.6888，混合模型 0.8979/0.6851。参数只用训练集内部验证集选择，测试集按每位用户的时间顺序留出较晚评分。

python main.py --evaluate 会在同一测试集上比较空白、偏差、UserCF、ItemCF 和混合模型，并保存阴性/阳性对照、加法/减法实验以及按用户 bootstrap 区间。
