import org.apache.commons.math3.linear.* ;
import java.util.Arrays ;

class SVMFit {

  double eps = 0.000001 ;
  int numChanged ;
  int examineAll ;
  RealMatrix x ;
  double[] y ;
  Double[] lambda ;
  RealVector w ;
  Double[] errorCache ;
  double alpha ;
  double C ;
  int criticalLambda ;
  double b ;
  double tol = 0.02 ;

  public SVMFit(RealMatrix x, double[] y, double C) {
    //initialized threshhold to zero
    b = 0 ;

    criticalLambda = 0 ;
    numChanged = 0 ;
    examineAll = 1 ;
    this.x = x ;
    this.y = y ;
    this.C = C;

    //initialize lambda array to zero
    lambda = new Double[y.length] ;
    for (int i=0 ; i< lambda.length; i++ ) {
      lambda[i] = 0d ; 
    }

    w = new ArrayRealVector(x.getColumnDimension());
    
    for (int i=0 ; i< x.getColumnDimension(); i++ ) {
      w.setEntry(i,   Math.random()) ; 
    }
    
    errorCache = new Double[y.length] ;    

    updateErrorCache() ;

  }


  private Double evalDualObjFunc(int replaceIdx, double replaceVal) {
    double temp = lambda[replaceIdx] ;
    lambda[replaceIdx] = replaceVal ;
    double out = 0 ;
    for (int  i=0 ; i < lambda.length ; i++) {
      out += lambda[i] ;
      for (int  j=0 ; j < lambda.length ; j++) {
        out -= (0.5 * (y[i] * y[j] ) * Kernels.linearKernel(x.getRowVector(i), x.getRowVector(j) )  * lambda[i] * lambda[j] ) ;
      }
    }
    lambda[replaceIdx] = temp ;
    return out ;
  }

  private int takeStep(int i1, int  i2,  double y2, double lambda2) {
    if (i1 == i2) return 0 ;
    double lambda1 = lambda[i1] ;
    double y1 = y[i1] ;
    double E1 = svmScoreError(i1);
    double E2 = svmScoreError(i2);
    double s = y1*y2 ;
    //Compute L, H
    double L, H ;
    if (Math.abs(y2 -y1) > 0.001) {
      L = Math.max(0, (lambda2 - lambda1)) ;
      H = Math.min(C, ( C + (lambda2 - lambda1))) ; 
    } else {
      L = Math.max(0, (lambda2 + lambda1 - C)) ;
      H = Math.min(C, (lambda2 + lambda1 )) ; 
    }
    if (((L -  H) > -eps) && ((L - H ) < eps) ) return 0 ;
    double k11 = Kernels.linearKernel(x.getRowVector(i1), x.getRowVector(i1));
    double k12 = Kernels.linearKernel(x.getRowVector(i1), x.getRowVector(i2));
    double k22 = Kernels.linearKernel(x.getRowVector(i2), x.getRowVector(i2));
    double eta = 2*k12-k11-k22 ;
    double nextLambda2 ;
    if (eta < 0) {
      nextLambda2 = lambda2 - y2*(E1-E2)/eta ;
      if (nextLambda2 < L) nextLambda2 = L ;
      else if (nextLambda2 > H) nextLambda2 = H;
    } else {
      double Lobj = evalDualObjFunc(i2 , L) ;
      double Hobj = evalDualObjFunc(i2, H) ; 
      if (Lobj > Hobj+eps) nextLambda2 = L;
      else if (Lobj < Hobj-eps)  nextLambda2 = H;
      else nextLambda2 = lambda2;
    }
    if (nextLambda2 < eps) nextLambda2 = 0d ;
    else if (nextLambda2 > (C-1e-8) ) nextLambda2 = C ;
    if (Math.abs(nextLambda2-lambda2) < (eps*(nextLambda2+lambda2+eps))) return 0 ;
    double nextLambda1 = lambda1+s*(lambda2-nextLambda2) ;
    //Update threshold to reflect change in Lagrange multipliers
    double b1 = E1 + (y1 * (nextLambda1 - lambda1)*k11) + (y2*(nextLambda2 - lambda2)*k12) + b ;
    double b2 = E2 + (y1 * (nextLambda1 - lambda1)*k12) + (y2*(nextLambda2 - lambda2)*k22) + b ;
    if ((nextLambda1 == 0 ) || (nextLambda1 == C )) {
      b = b1 ;
    } else if ((nextLambda2 == 0 ) || (nextLambda2 == C )) {
      b = b2 ;
    } else {
      b = (b1+b2) / 2 ;
    }
    //Update weight vector to reflect change in a1 & a2, if linear SVM
    lambda[i1] = nextLambda1; 
    lambda[i2] = nextLambda2; 
    //Update error cache using new Lagrange multipliers
    updateErrorCache();    
    //Store a1 in the alpha array
    lambda[i1] = nextLambda1 ; 
    //Store a2 in the alpha array
    lambda[i2] = nextLambda2 ; 
    if ((nextLambda1 != 0) || (nextLambda1 != C ) ) criticalLambda++ ; 
    if ((nextLambda2 != 0) || (nextLambda2 != C ) ) criticalLambda++ ; 
    if ((lambda1 != 0) || (lambda1 != C ) ) criticalLambda-- ; 
    if ((lambda2 != 0) || (lambda2 != C ) ) criticalLambda-- ; 
  
    return 1 ;
  }

  private void updateErrorCache() {
    for (int i=0 ; i < y.length; i++) {
      Double k = Kernels.linearKernel(w, x.getRowVector(i) );
      double error =  -1 * (y[i] * (k - b) - 1) ;
      errorCache[i] = error ; 
    }
  }

  private Double svmScoreError(int i) {
    if ( errorCache[i] != null  ) return errorCache[i];
    Double k = Kernels.linearKernel(w, x.getRowVector(i) );
    double error =  -1 * (y[i] * (k - b) - 1) ;
    errorCache[i] = error ; 
    return errorCache[i] ;
  }

  private int secondChoiceHeuristic(int exampleId, Double errVal) {
    int minId = -1 ;
    int maxId = -1 ;
    double minError = 999999 ;
    double maxError = -999999 ;
    for (int i=0 ; i < y.length; i++) {
      if ((errorCache[i] == null) || (i==exampleId)) continue ;
      if (errorCache[i] > maxError) {
        maxError = errorCache[i] ;
        minId = i ;
      }
      if (errorCache[i] < minError) {
        minError = errorCache[i] ; 
        maxId = i ;
      }
    }
    if (errVal > 0) {
      //return max error
      return maxId ;
    } else {
      //return min error
      return minId ;
    }
  }

  private int examineExample(int i2) {
    double y2 = y[i2] ;
    double lambda2 = lambda[i2] ;
    double error2 = svmScoreError(i2) ;
    int i1 ; 
    Double r2 = error2*y2 ;

    System.out.println("Examining example " + i2 );
    System.out.println("\t r2: " + r2 );
    System.out.println("\t lambda2 " + lambda2 );
    System.out.println("\t error2 " + error2 );
    System.out.println("\t y2 " + y2 );

    if (((r2 < -tol) && (lambda2 < C)) || ((r2 > tol) && (lambda2 > 0))) {
      System.out.println("\tkkt violated "  );
      if (criticalLambda > 1) {
        i1 = secondChoiceHeuristic(i2, error2) ;
        if (takeStep(i1, i2, y2, lambda2 ) == 1 ) return 1 ; 
      }
      int startIdx = 0 + (int)(Math.random() * (( lambda.length -1  ) + 1)) ;
      int currIdx = -1 ;
      //loop over all non-zero and non-C alpha, starting at random point
      while (currIdx != startIdx ) {
        if (currIdx == -1) currIdx = startIdx ;
        if ((lambda[currIdx].equals(0)) || (lambda[currIdx].equals(C)))  continue ;
        i1 = currIdx ; 
        if (takeStep(i1,i2, y2, lambda2 ) == 1) return 1 ;
        currIdx++ ;
        if (currIdx >= lambda.length) currIdx = 0 ;
      }
      //loop over all possible i1, starting at a random point
      startIdx = 0 + (int)(Math.random() * (( lambda.length -1  ) + 1)) ;
      currIdx = -1 ;
      while (currIdx != startIdx ) {
        if (currIdx == -1) currIdx = startIdx ;
        i1 = currIdx ;
        if (takeStep(i1, i2, y2, lambda2) == 1) return 1;
        if (currIdx >= lambda.length) currIdx = 0 ;
      }
    }
    System.out.println("\tno change for example");
    return 0;
  }



  public  RealVector solveForW(  ) {
    // looping over training examples in violation OR if examine all is tripped.
    while (numChanged > 0 || examineAll == 1) {
      System.out.println("entering training loop: numchanged: " + numChanged + " examineAll " + examineAll) ;
      numChanged = 0;
      if (examineAll == 1) {
        //loop I over all training examples
        for (int i = 0; i < y.length ; i++ ) {
          numChanged += examineExample(i) ;
        }
      } else {
        //loop I over examples where lambda is not 0 & not C
        for (int i = 0; i < y.length ; i++ ) {
          if ((lambda[i] < 0.0001)  &&  (lambda[i] > -0.0001 ) ) continue ;
          if ((lambda[i] - C < 0.0001)  &&  (lambda[i] - C > -0.0001 ) ) continue ;
          numChanged += examineExample(i) ;
        }
      }

      if (examineAll == 1) {
        examineAll = 0;
      } else if (numChanged == 0) {
        examineAll  = 1 ;
      }
      System.out.println("leaving training loop: numchanged: " + numChanged + " examineAll " + examineAll) ;
    }
    return w ;
  }

  public double[] scoreExamples( RealMatrix x_test ) {
    double[] out = new double[x_test.getRowDimension()] ;  
    for (int i=0 ; i < out.length ; i++ ) {
      double k = Kernels.linearKernel(w, x_test.getRowVector(i) );
      out[i] = k - b ; 
    }
    return out ;
  }

  public static void main(String[] args) {
    double c =  10 ;
    double[] y_example = {1d,1d,1d,1d,-1d,-1d, -1d, -1d   } ;
    double[][] matrixData = { {0d,0d}, {1d,0d},{0.2d, 0.15d}, {0.1d , 0.1d},  {0d,1d }, {1d, 1d}, {0.7d, 0.9d}, {0.3d, 0.8d}    };
    RealMatrix x_example = MatrixUtils.createRealMatrix(matrixData);
    
    SVMFit model = new SVMFit( x_example  ,y_example,  c) ;
    RealVector w = model.solveForW();
    System.out.println(Arrays.toString(w.toArray()));
      
    double[] test = model.scoreExamples(x_example ) ; 
    System.out.println(Arrays.toString(test));

 
  }


}
