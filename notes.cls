\LoadClass[UTF8]{ctexart}
\NeedsTeXFormat{LaTeX2e}

\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{geometry}
\RequirePackage{tikz}
\RequirePackage{pgfplots}
\RequirePackage{booktabs}
\RequirePackage{subfigure}
\RequirePackage{enumitem}
\RequirePackage{hyperref}
\RequirePackage{color}

\linespread{1.5}

\hypersetup{colorlinks}

\savegeometry{standard}
\newgeometry{
    top=1in,
    bottom=1in,
    left=1.25in,
    right=1.25in,
}
\savegeometry{word-normal}
\newgeometry{
    top=0.5in,
    bottom=0.5in,
    left=0.5in,
    right=0.5in,
}
\savegeometry{word-narrow}
\newgeometry{
    top=1in,
    bottom=1in,
    left=0.75in,
    right=0.75in,
}
\savegeometry{word-moderate}
\newgeometry{
    top=1in,
    bottom=1in,
    left=2in,
    right=2in,
}
\savegeometry{word-wide}
\loadgeometry{standard}

\author{皇甫硕龙}

\newcommand{\separate}[1][0.5pt]{\rule{0.25\textwidth}{0pt}\rule{0.5\textwidth}{#1}}

\newcommand{\N}{\mathbb N}
\newcommand{\R}{\mathbb R}
\newcommand{\Z}{\mathbb Z}
\newcommand{\C}{\mathbb C}
\newcommand{\Q}{\mathbb Q}
\newcommand{\dd}{\mathrm d}
\newcommand{\Cov}{\mathrm {Cov}}
\renewcommand{\geq}{\geqslant}
\renewcommand{\leq}{\leqslant}
\newcommand{\eqnref}[1]{(\ref{#1})}

% 对角矩阵
\newcommand{\diag}{\mathrm{diag}}

% 概率
\newcommand{\prob}[1]{P\left(#1\right)}
\newcommand{\cprob}[2]{P\left(#1|#2\right)}

% 期望
\newcommand{\expt}[1]{E\left(#1\right)}
\newcommand{\cexpt}[2]{E\left(#1|#2\right)}

% 随机分布
\newcommand{\poisson}[2]{\frac{{#1}^{#2}}{(#2)!}e^{-#1}}

\newcommand{\bs}{\boldsymbol}

\newenvironment{subquestions}{
    \begin{enumerate}[label=\textbf{(\alph*)}, parsep=3pt, itemsep=3pt]
}{
    \end{enumerate}
}

\newenvironment{derive}[1][]{
    \begin{equation}
        \begin{aligned}
            #1
}{
        \end{aligned}
    \end{equation}
}

\newenvironment{derive*}[1][]{
    \begin{equation*}
        \begin{aligned}
            #1
}{
        \end{aligned}
    \end{equation*}
}

\newtheorem{definition}{定义}[subsection]
\newtheorem{proposition}{命题}[subsection]
\newtheorem{proof}{证明}[subsection]
\newtheorem{lemma}{引理}[subsection]
\newtheorem{example}{例}[subsection]

% Tikz settings

% \pgfplotsset{

% }