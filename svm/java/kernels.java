import org.apache.commons.math3.linear.* ;

class Kernels {

  public static double guassianRadialBasisKernel( RealVector x, RealVector y   ) {
    // a gaussian radial basis kernel
    return Math.exp( -0.5 * Math.pow(x.getL1Distance(y), 2  )) ;
  }

  public static double homogenousPolynomialKernel( RealVector x, RealVector y  ) {
   // a homogenous polynomial kernel of degree 2 
    return Math.pow(x.dotProduct(y), 2);
  }

  public static double linearKernel(RealVector x, RealVector y) {
    return x.dotProduct(y) ;
  }

}
