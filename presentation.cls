\LoadClass[aspectratio=169]{beamer}
\NeedsTeXFormat{LaTeX2e}

\RequirePackage{amsmath}
\RequirePackage{setspace}
\RequirePackage{ctex}

\linespread{1.5}

\hypersetup{colorlinks}

\author{皇甫硕龙}

\newcommand{\separate}[1][0.5pt]{\rule{0.25\textwidth}{0pt}\rule{0.5\textwidth}{#1}}

\newcommand{\N}{\mathbb N}
\newcommand{\R}{\mathbb R}
\newcommand{\Z}{\mathbb Z}
\newcommand{\C}{\mathbb C}
\newcommand{\Q}{\mathbb Q}
\newcommand{\dd}{\mathrm d}
\renewcommand{\geq}{\geqslant}
\renewcommand{\leq}{\leqslant}
\newcommand{\eqnref}[1]{(\ref{#1})}

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