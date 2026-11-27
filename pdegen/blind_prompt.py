"""Render standalone, label-free prompts from the canonical PDE specifications."""
from fractions import Fraction


INTRO = """请独立求解下面的 PDE 谱稳定性题。本轮不要联网、查找现成解答或运行数值程序。可以引用标准定理，请说明其假设为何在本题成立。"""
QUESTION = r"""问题：这个线性化算子在开右半平面 \(\{\lambda:\Re\lambda>0\}\) 中是否有谱？

请给出明确结论和完整证明。证明失稳只需严格建立一个位于该区域的谱点，证明稳定需要排除该区域内的全部谱。若无法完成，请明确回答「未能确定」，并区分已经证明的结论和卡住的具体步骤。"""


def fraction_tex(value):
    number = Fraction(value)
    if number.denominator == 1:
        return str(number.numerator)
    prefix = "-" if number < 0 else ""
    return prefix + rf"\frac{{{abs(number.numerator)}}}{{{number.denominator}}}"


def a1_prompt(parameters):
    q, a, beta = (fraction_tex(parameters[name]) for name in ("q", "a", "beta"))
    return r"""在整条实线上，令 \(X=(u,y,v,w)^\top\)，考虑
\[
X_t=DX_{xx}-AX_x+F(X)+\partial_xG(X)
\]
以及参数 \(q=@q@,\ a=@a@,\ \beta=@beta@\)。取
\[
D=\operatorname{diag}(1,3,1,1+q/2),\qquad
A=\operatorname{diag}(1+q/2,1,1-q/2,1+q).
\]
令 \(R=v-u^2,\ S=w-u^3,\ P_j(u)=j^2u^j-j(j+1)u^{j+2}\)，并取
\[
F(X)=\begin{pmatrix}
-P_1(u)+qy/2+qu(3R+2S)\\
-3y+\beta u^2R\\
-P_2(u)-quy+(6u^2-a)R-2S\\
-(1+q/2)P_3(u)+3qu^2y+(6u^2-a+qu)S+2R
\end{pmatrix},
\qquad
G(X)=\begin{pmatrix}0\\6u^3\\0\\0\end{pmatrix}.
\]
指定的脉冲速度为 \(1\)。在随波坐标 \(z=x-t\) 中，令 \(s(z)=\operatorname{sech}z\)，指定剖面为
\[
X_*(z)=(s,s',s^2,s^3)^\top.
\]
使用复值无权空间 \(L^2(\mathbb R;\mathbb C^4)\) 上的随波线性化，定义域为 \(H^2(\mathbb R;\mathbb C^4)\)。
""".replace("@q@", q).replace("@a@", a).replace("@beta@", beta).strip()


def d2_prompt(parameters):
    width = Fraction(parameters["width"])
    start = Fraction(parameters["tau"])
    end = fraction_tex(start + width)
    gain = fraction_tex(Fraction(parameters["sigma"]) / width)
    epsilon = fraction_tex(parameters["epsilon"])
    return r"""在整条实线上考虑
\[
u_t=u_{xx}+u-u^3,\qquad
v_t=v_{xx}-v-@gain@(1-u^2)(1+\varepsilon u)
\int_{@start@}^{@end@}v(x,t-s)\,ds,\qquad \varepsilon=@epsilon@.
\]
指定前沿为 \((u_*,v_*)=(\tanh(x/\sqrt2),0)\)。在该前沿处线性化，使用复值无权空间
\[
L^2(\mathbb R;\mathbb C)\times C([-@end@,0];L^2(\mathbb R;\mathbb C)).
\]
第一因子是 \(u\) 扰动，定义域为 \(H^2(\mathbb R;\mathbb C)\)。第二因子是 \(v\) 的历史扰动。历史生成元在内部对历史坐标求导，定义域由 \(C^1\) 历史组成，要求零时刻端点值属于 \(H^2\)，且端点导数满足线性化 \(v\) 方程。
""".replace("@start@", fraction_tex(start)).replace("@end@", end).replace(
        "@gain@", gain).replace("@epsilon@", epsilon).strip()


def e3_prompt(parameters):
    gain = fraction_tex(parameters["gain"])
    return r"""在移动半直线 \(x<s(t)\) 上，考虑
\[
u_t=u_{xx}-u+v-u^2,\qquad v_t=v_{xx}-\tfrac52v,\qquad
u(s(t),t)=v(s(t),t)=1.
\]
界面和两个反馈状态满足
\[
s'=\tfrac32+a,\qquad a'=b,\qquad
b'=-a-\tfrac14b+g\,[u_x(s(t)^-,t)-\tfrac12+v_x(s(t)^-,t)-1],
\qquad g=@gain@.
\]
指定的推进波为 \(s_*(t)=3t/2\)、\(a_*=b_*=0\)。在附着于界面的坐标 \(z=x-s(t)\le0\) 下，剖面为
\[
U(z)=e^{z/2},\qquad V(z)=e^z.
\]
在该波处线性化，使用无权空间 \(L^2((-\infty,0);\mathbb C^2)\times\mathbb C^3\)。两个场扰动的定义域为 \(H^2\cap H^1_0\)，三个标量分别表示界面位移和两个反馈状态的扰动。
""".replace("@gain@", gain).strip()



def b3_prompt(parameters):
    return r"""在整条实线上，令 \(X\in\mathbb R^2\)，考虑
\[
X_t=D(X_{xx}+X)+(1-|X|^2)(bI+\chi J)X,\qquad
D=\operatorname{diag}(1,2),\quad J=\begin{pmatrix}0&-1\\1&0\end{pmatrix},
\qquad b=@damping@,\quad\chi=@skew@.
\]
指定静止波为 \(X_*(x)=(\cos x,\sin x)^\top\)。使用复值无权空间
\(L^2(\mathbb R;\mathbb C^2)\) 上的线性化，定义域为 \(H^2(\mathbb R;\mathbb C^2)\)。
""".replace("@damping@", fraction_tex(parameters["damping"])).replace(
        "@skew@", fraction_tex(parameters["skew"])).strip()


def c2_prompt(parameters):
    return r"""在整条实线上，令 \(\psi,\chi\) 为复场，考虑
\[
\begin{aligned}
i\psi_t&=-\psi_{xx}+\psi-|\psi|^2\psi-\alpha|\chi|^2\psi-\beta\chi^2\overline\psi,\\
i\chi_t&=-\chi_{xx}+\chi-\alpha|\psi|^2\chi-\beta\psi^2\overline\chi,
\qquad \alpha=@alpha@,\quad\beta=@beta@.
\end{aligned}
\]
指定静止波为 \((\psi_*,\chi_*)=(\sqrt2\operatorname{sech} x,0)\)。
取四个实分量的线性化，再复化到无权空间 \(L^2(\mathbb R;\mathbb C^4)\)，定义域为 \(H^2(\mathbb R;\mathbb C^4)\)。
""".replace("@alpha@", fraction_tex(parameters["alpha"])).replace(
        "@beta@", fraction_tex(parameters["beta"])).strip()


def f3_prompt(parameters):
    return r"""在整条实线上，令 \(f(x)=2+\cos x\)、\(R(t)\) 为平面上转角为 \(t\) 的旋转矩阵，
\(J=R'(0)\)。设 \(X_*(x,t)=f(x)(\cos t,\sin t)^\top\)，考虑
\[
X_t=X_{xx}-(f''/f)X+JX-\tfrac14(X-X_*)+
gR(t)S(x,t)R(t)^\top(X-X_*),\qquad g=@gain@.
\]
矩阵 \(S\) 的时间周期为 \(2\pi\)。在 \(0\le t<\pi\) 它只有 \(S_{12}=1/2+\cos x\) 非零，
在 \(\pi\le t<2\pi\) 它只有 \(S_{21}=1/2+\sin x\) 非零。
取指定轨道 \(X_*\) 的线性化，在无权空间 \(L^2(\mathbb R;\mathbb C^2)\) 中把两段演化连续拼接。
单周期演化算子将 \(t=0\) 的扰动送到 \(t=2\pi\)。
""".replace("@gain@", fraction_tex(parameters["gain"])).strip()


def f4_prompt(parameters):
    return r"""在整条实线上，令 \(X(x,t)\in\mathbb R^2\)，
\[
f(x)=2+\cos x+\tfrac14\sin(2x),\qquad
r(t)=(\cos(2t),\sin(2t))^\top.
\]
考虑
\[
X_t=X_{xx}-\frac{f''(x)}{f(x)}X+f(x)r'(t)+B(t)\bigl(X-f(x)r(t)\bigr),
\qquad
B(t)=\begin{pmatrix}-1/8&1\\-k(t)&-1/8\end{pmatrix},
\]
其中
\[
k(t)=a+h\bigl(\cos(2t)+\tfrac14\sin(4t)\bigr),
\qquad a=@stiffness@,\quad h=@modulation@.
\]
指定的周期轨道为 \(X_*(x,t)=f(x)r(t)\)。在该轨道处取线性化，
使用无权空间 \(L^2(\mathbb R;\mathbb C^2)\)，空间算子的定义域为 \(H^2(\mathbb R;\mathbb C^2)\)。
单周期演化算子将 \(t=0\) 的扰动送到 \(t=\pi\)。
""".replace("@stiffness@", fraction_tex(parameters["stiffness"])).replace(
        "@modulation@", fraction_tex(parameters["modulation"])).strip()


FLOQUET_QUESTION = """问题：这个单周期演化算子在单位圆外是否有谱？

请给出明确结论和完整证明。证明失稳只需严格建立一个单位圆外的谱点，证明稳定需要排除单位圆外的全部谱。若无法完成，请明确回答「未能确定」，并区分已经证明的结论和卡住的具体步骤。"""


RENDERERS = {"A1": a1_prompt, "D2": d2_prompt, "E3": e3_prompt,
             "B3": b3_prompt, "C2": c2_prompt, "F3": f3_prompt, "F4": f4_prompt}


def render_problem(item):
    family = item["family_id"]
    question = FLOQUET_QUESTION if family in ("F3", "F4") else QUESTION
    return "\n\n".join((INTRO, RENDERERS[family](item["parameters"]), question)) + "\n"
