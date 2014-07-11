import org.apache.commons.math3.linear.* ;


class RegressionTools {

  public static RealMatrix solveForCoefficients(RealMatrix x, RealMatrix y) {
    return (new LUDecomposition(    x.transpose().multiply(x)      )
        .getSolver()
        .getInverse())
      .multiply(x.transpose())
      .multiply(y)   ;
  }

}
