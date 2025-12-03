import math
import numpy as np

ObservedCalls = 180
TotalHours = 10
PriorAlpha = 18.0
PriorBeta = 1.0
HdiMass = 0.94

AlphaPost = PriorAlpha + ObservedCalls
BetaPost = PriorBeta + TotalHours

PostMean = AlphaPost / BetaPost
PostVar = AlphaPost / (BetaPost ** 2)
PostStd = math.sqrt(PostVar)
PostMode = (AlphaPost - 1) / BetaPost if AlphaPost > 1 else 0.0

def Log_Gamma_Pdf_Rate(LambdaValue, Alpha, Beta):
    if LambdaValue <= 0:
        return -math.inf
    return Alpha*math.log(Beta) + (Alpha-1)*math.log(LambdaValue) - Beta*LambdaValue - math.lgamma(Alpha)

Lo = max(1e-9, PostMean - 8.0 * PostStd)
Hi = PostMean + 8.0 * PostStd
Grid = np.linspace(Lo, Hi, 4000)

LogPdf = AlphaPost*np.log(BetaPost) + (AlphaPost-1)*np.log(Grid) - BetaPost*Grid - math.lgamma(AlphaPost)
LogPdfMax = LogPdf.max()
Pdf = np.exp(LogPdf - LogPdfMax)
Area = np.trapezoid(Pdf, Grid)
Pdf = Pdf / Area

IdxSorted = np.argsort(-Pdf)
Dx = Grid[1] - Grid[0]
CumMass = 0.0
Included = np.zeros_like(Pdf, dtype=bool)
for i in IdxSorted:
    Included[i] = True
    CumMass += Pdf[i] * Dx
    if CumMass >= HdiMass:
        break

Where = np.where(Included)[0]
HdiLower = Grid[Where.min()]
HdiUpper = Grid[Where.max()]

print("Posterior: Gamma(shape=AlphaPost, rate=BetaPost)")
print(f"  AlphaPost = {AlphaPost:.6g},  BetaPost = {BetaPost:.6g}")
print(f"  Mean      = {PostMean:.6f}")
print(f"  Std       = {PostStd:.6f}")
print(f"  Mode      = {PostMode:.6f}")
print(f"  {int(HdiMass*100)}% HDI ≈ [{HdiLower:.6f}, {HdiUpper:.6f}]")
