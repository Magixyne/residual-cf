# MovieLens 协同过滤实验报告

- report.tex：报告 LaTeX 源码。
- report.pdf：由 report.tex 编译生成的报告。
- figures/：评分分布/模型误差与内存/阻尼示意图。

正文、公式、表格和参考文献由 LaTeX 排布；附录不再使用代码截图，而是用 listings 直接引入仓库根目录的 `recommender.py` 代码节选，避免图文与源码不一致。

## 编译

需要 XeLaTeX（或 Tectonic）、中文字体 Microsoft YaHei 与等宽字体 Consolas。进入本目录后运行：

    xelatex report.tex
    xelatex report.tex

若系统没有 Microsoft YaHei/Consolas，可把 report.tex 中的 `\setmainfont`、`\setsansfont`、`\setmonofont` 替换为本机可用字体。
