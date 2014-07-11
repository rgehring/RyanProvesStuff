import java.io.FileReader;
import java.io.Reader;
import java.io.IOException ;
import org.apache.commons.cli.*;
import org.apache.commons.csv.*;
import org.apache.commons.math3.*;
import org.apache.commons.math3.linear.* ;

class RunRegression {

  public static void main(String[] args) throws IOException, ParseException {
    
    // create Options object
    Options options = new Options();

    // add t option
    options.addOption("i", true, "input file (excel csv expected)");
    options.addOption("y", true, "name of response variable column header");
    
    CommandLineParser parser = new GnuParser();
    CommandLine cmd = parser.parse( options, args);

    String pathToData = cmd.getOptionValue("i");
    String responseName = cmd.getOptionValue("y");

    if((pathToData == null  ) || (responseName == null)) {
      System.out.println("missing input csv. correct call is java RunRegression -i <filepath> -y <response_column_header>");
      System.exit(1) ;
    }

    // read the input data
    Reader in = new FileReader(pathToData);
    
    //determine number rows n cols
    int numRows = 0 ;
    int numCols = 0 ;
    Iterable<CSVRecord> records = CSVFormat.EXCEL.parse(in);
    
    String[] xHeaders = new String[0];
    int responseIdx =-1;

    for (CSVRecord record : records) {
      numRows++ ;
      if (numRows == 1) {
        numCols = record.size() ;
        xHeaders = new String[numCols - 1] ; 
        for (int i=0; i< record.size() ; i++ ) {
          if (record.get(i).equals(responseName)) {
            responseIdx = i ;
          } else {
            if (responseIdx == -1) xHeaders[i] = record.get(i) ;
            else xHeaders[i-1] = record.get(i) ;
          }
        }
        if (responseIdx==-1) System.exit(1) ;
      } 
    }
    numRows--;
    System.out.println("Response index is " + responseIdx);

    // initialize array to load data
    double[][] xdata = new double[numRows][numCols-1];
    double[][] ydata = new double[numRows][1];

    in.close(); 
    in = new FileReader(pathToData);
    records = CSVFormat.EXCEL.withHeader().parse(in);
    int i=0 ;
    for (CSVRecord record : records) {
      for (int j=0 ; j < record.size() ; j++) {
        // add record to real matrix data
        if (j == responseIdx) {
          ydata[i][0] = Double.parseDouble(record.get(j) );
        } else {
          if (j < responseIdx) { 
            xdata[i][ j] =  Double.parseDouble(record.get(j)) ;
          } else {
            xdata[i][ j-1] =  Double.parseDouble(record.get(j)) ;
          }
        }
      }
      i++ ;
    } 

    // print x matrix
    System.out.println("====================");
    System.out.println("      X    ");
    System.out.println("====================");
    for (double[] row : xdata) {
      for (double val : row) {
        System.out.print(val+"\t");
      }
        System.out.print("\n");
    }
    
    // print y matrix
    System.out.println("====================");
    System.out.println("      Y    ");
    System.out.println("====================");
    for (double[] row : ydata) {
      for (double val : row) {
        System.out.print(val+"\t");
      }
        System.out.print("\n");
    }

    //the matrix
    RealMatrix x = MatrixUtils.createRealMatrix(xdata);
    RealMatrix y = MatrixUtils.createRealMatrix(ydata);

    // print some summary stats
    System.out.println(numRows + " Rows by " + numCols + " cols.");

    // Invert p, using LU decomposition
    RealMatrix estimates = RegressionTools.solveForCoefficients(x,y) ;

    // print the coefficient estimates
    double[] est = estimates.getColumn(0) ;
    System.out.println("=======================");
    System.out.println("Coefficient Estimates");
    System.out.println("=======================");
    for (int j=0; j < est.length ; j++) {
      System.out.println(xHeaders[j]+":\t"+est[j]) ;
    }

  }

}
