import java.io.File;
import java.io.FileNotFoundException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.Scanner;

public class ProjectTwo {

	// Class Properties:
	ArrayList<Matrix> matrixClass1 = new ArrayList<>();
	Matrix meanClass1;
	Matrix covarianceClass1;
	double trace1;
	double determinant1;
	
	public static void main(String[] args) {
		ProjectTwo p2 = new ProjectTwo();
	}
	
	
	public ProjectTwo() {
		
		// Read in data from text file
		ArrayList<String> temp = new ArrayList<>();
		Scanner input = null;
		File infile = new File("Resource/test2.txt");
		
		try {
			input = new Scanner(infile, "UTF-8");
		} catch (FileNotFoundException e) {
			System.out.println("File not found.");
			e.printStackTrace();
		}
		
		while (input.hasNextLine()) {
			temp.add(input.nextLine());
		}
		
		input.close();
		
		int lineCounter = 0;
		
		for (String ln: temp) {
			if (lineCounter >= 1) {
				
				@SuppressWarnings("resource")
				Scanner lineScanner = new Scanner(ln);
        			lineScanner.useDelimiter(";");
        			
        			// Create Class 1 2x1 matrix and add to ArrayList
        			double[][] vectorA = new double[2][1];
        			vectorA[0][0] = Double.parseDouble(lineScanner.next().trim());
        			vectorA[1][0] = Double.parseDouble(lineScanner.next().trim());
        			Matrix matrixA = new Matrix(vectorA);
        			this.matrixClass1.add(matrixA);
        			
        			lineCounter += 1;
			}
			lineCounter += 1;
		}
		
		// Problem 1_i: Compute the mean vector for eigendata:
		this.meanClass1 = computeMean(matrixClass1);
		
		System.out.println("The Mean of Eigendata is: " + "x = " + meanClass1.getElement(0, 0) + "     " + "y = " + meanClass1.getElement(1, 0));
		
		// Problem 1_i: Compute the covariance matrix for eigendata:
		this.covarianceClass1 = computeCovariance(matrixClass1, meanClass1);
		
		System.out.println("The Covariance Matrix of Eigendata is: " + "\n" + covarianceClass1.getElement(0, 0) + "   " + covarianceClass1.getElement(0, 1) + "\n" + covarianceClass1.getElement(1, 0) + "   " + covarianceClass1.getElement(1, 1) + "\n");
	
		// Problem 1_ii: Compute the trace for covariance matrix from eigendata:
		this.trace1 = calculateTrace(covarianceClass1);
		
		System.out.println("The Trace of the Covariance Matrix is: " + trace1);
		
		// Problem 1_iii: Compute the determinant from eigendata: 
		this.determinant1 = gaussReductionDeterminant(covarianceClass1);
		
		System.out.println("The Determinant of Covariance Matrix is: " + determinant1);
		
		// Problem 1_iv: Compute eigenvalues of the covariance matrix:
		getEigenValues(covarianceClass1);
		
		 
		double[][] monicPolynomial = {{-1.0, 73.0/6.0, 161.0/6.0, -1140.0/6.0, -1516.0/6.0, 1456.0/6.0},
									 {1, 0, 0, 0, 0, 0},
									 {0, 1, 0, 0, 0, 0},
									 {0, 0, 1, 0, 0, 0},
									 {0, 0, 0, 1, 0, 0},
									 {0, 0, 0, 0, 1, 0}};
		
		
		//double[][] monicPolynomial = {{-1,-5, 0, 20, 10, -2},{1,0,0,0,0,0},{0,1,0,0,0,0},{0,0,1,0,0,0}, {0,0,0,1,0,0},{0,0,0,0,1,0}};
	
		Matrix monicA = new Matrix(monicPolynomial.length, monicPolynomial[0].length);
		
		// Assign partA elements to Matrix firstM
		for (int i = 0; i < monicPolynomial.length; i++) {
			for (int j = 0; j < monicPolynomial[0].length; j++) {
				monicA.assignElement(monicPolynomial[i][j], i, j);
			}
		}
		System.out.println("The Characteristic Polynomial is: ");
		leverrier(monicA);
		
		double[][] practice = {{4,2,2,1}, {2,-3,1,1}, {2,1,3,1}, {1,1,1,2}};
		Matrix prac = new Matrix(practice.length, practice[0].length);
		// Assign partA elements to Matrix firstM
		for (int i = 0; i < practice.length; i++) {
			for (int j = 0; j < practice[0].length; j++) {
				prac.assignElement(practice[i][j], i, j);
			}
		}
		
		//double[][] practice2 = {{0,1,0,0,0},{1,0,0,1,0},{0,0,0,1,1},{0,1,1,0,1},{0,0,1,1,0}};
		double[][] practice2 = {{1,1,1},{1,2,3}, {1,3,6}};
		Matrix prac2 = new Matrix(practice2.length, practice2[0].length);
		
		for (int i = 0; i < practice2.length; i++) {
			for (int j = 0; j < practice2[0].length; j++) {
				prac2.assignElement(practice2[i][j], i, j);
			}
		}
		
		
		double dP = powerMethod(prac2);
		System.out.println(dP);
		
		//Matrix A = householder(prac);
		//System.out.println(A);
		
		//travelingSalesmanProblem();
	
	}
	
	/**
	 * The computeMean method calculates the mean vector for each corresponding row and column entry. 
	 * @param mList		an ArrayList of matrices with equivalent dimensions. 
	 * @return meanMatrix	a Matrix containing the mean values for each row and column entry. 
	 */
	private Matrix computeMean(ArrayList<Matrix> mList) {
		
		double[][] sumArray = mList.get(0).getMatrix();
		double counter = 1;
		
		for (int i = 1; i < mList.size(); i++) {
			sumArray = add(sumArray, mList.get(i).getMatrix());
			counter += 1;
		}
		
		double[][] mean = matrixScalar(sumArray, counter);
		Matrix meanMatrix = new Matrix(mean);
		return meanMatrix;
	}
	
	/**
	 * The computeCovariance method calculates the covariance matrix for an ArrayList of matrices, by first subtracting the mean vector
	 * 		from each of the measurement vectors. Then multiply each resulting n-by-1 difference vector by its 1-by-n transpose, resulting in
	 * 		an n-by-n product. Lastly, compute the mean of the n-by-n products. 
	 * @param mList		an ArrayList of n-by-n matrices with equivalent dimensions
	 * @param meanMatrix		a Matrix containing the mean values for each row and column entry 
	 * @return covarianceMatrix     the covariance Matrix for an ArrayList of matrices
	 */
	private Matrix computeCovariance(ArrayList<Matrix> mList, Matrix meanMatrix) {
		
		ArrayList<Matrix> multiplyClass = new ArrayList<>();
		Matrix covarianceMatrix;
		
		for (int i = 0; i < mList.size(); i++) {
			
			Matrix diffMatrix = difference(mList.get(i), meanMatrix);
			Matrix transMatrix = transpose(diffMatrix);
			Matrix multMatrix = multiply(diffMatrix, transMatrix);
			multiplyClass.add(multMatrix);
		}
		
		covarianceMatrix = computeMean(multiplyClass);
		return covarianceMatrix;	
	}
	
	/**
	 * The add method performs the addition operation on two m-by-n matrices; they must have the same number
	 * 		of rows and columns.
	 * @param matrixA     an m-by-n matrix
	 * @param matrixB     an m-by-n matrix
	 * @return sumMatrix     an m-by-n matrix of corresponding elements equal to the sum of matrixA and matrixB
	 */
	private double[][] add(double[][] matrixA, double[][] matrixB) {
		
		double[][] sumMatrix = new double[matrixA.length][matrixA[0].length];

		if (matrixA.length == matrixB.length && matrixA[0].length == matrixB[0].length) {
			for (int i = 0; i < matrixA.length; i++) {
				for (int j = 0; j < matrixA[0].length; j++) {
					sumMatrix[i][j] = matrixA[i][j] + matrixB[i][j];
				}
			}
		}
		return sumMatrix;
	}
	
	/**
	 * 
	 * @param matrixA
	 * @param matrixB
	 * @return
	 */
	private Matrix add(Matrix matrixA, Matrix matrixB) {
		
		Matrix sumMatrix = new Matrix(matrixA.getRows(), matrixA.getColumns());
		
		if (matrixA.getRows() == matrixB.getRows() && matrixA.getColumns() == matrixB.getColumns()) {
			for (int i = 0; i < matrixA.getRows(); i++) {
				for (int j = 0; j < matrixA.getColumns(); j++) {
					sumMatrix.assignElement(matrixA.getElement(i, j) + matrixB.getElement(i, j), i, j);
				}
			}
		}
		return sumMatrix;
	}
	
	/**
	 * The matrixScalar methods multiplies each row and column entry of an m-by-n matrix, by a scalar multiple. 
	 * @param matrixA     an m-by-n matrix
	 * @param scalar     a (double) scalar multiple  
	 * @return matrixA     an m-by-n matrix of corresponding elements equal to product of each element and a scalar
	 */
	private double[][] matrixScalar(double[][] matrixA, double scalar) {
		
		for (int i = 0; i < matrixA.length; i++) {
			for (int j = 0; j < matrixA[0].length; j++) {
				matrixA[i][j] = matrixA[i][j] / scalar;
			}
		}
		return matrixA;
	}
	
	/**
	 * 
	 * @param matrixA
	 * @param scalar
	 * @return
	 */
	private Matrix matrixScalar(Matrix matrixA, double scalar) {
		
		for (int i = 0; i < matrixA.getRows(); i++) {
			for (int j = 0; j < matrixA.getColumns(); j++) {
				matrixA.assignElement(matrixA.getElement(i, j) * scalar, i, j);
			}
		}
		return matrixA;
	}
	
	/**
	 * The transpose method generates a new matrix whose columns are the rows of the original, and whose rows 
	 *      are the columns of the original. 
	 * @param matrixA     an m-by-n matrix
	 * @return transposeMatrix     an n-by-m matrix 
	 */
	private Matrix transpose(Matrix matrixA) {
		
		Matrix transposeMatrix = new Matrix(matrixA.getColumns(), matrixA.getRows());
	
		for (int j = 0; j < matrixA.getRows(); j++) {
			for (int k = 0; k < matrixA.getColumns(); k++) {
				transposeMatrix.assignElement(matrixA.getElement(j, k), k, j);
			}
		}
		return transposeMatrix;
	}
	
	/**
	 * The multiply method performs multiplication operation on two matrices; columns of first matrix must 
	 *      equal rows of second matrix.
	 * @param matrixA     an m-by-n matrix
	 * @param matrixB     an n-by-p matrix
	 * @return multiplyMatrix     an m-by-p matrix of corresponding elements equal to the product of matrixA and matrixB
	 */
	private Matrix multiply(Matrix matrixA, Matrix matrixB) {
		
		Matrix multiplyMatrix = new Matrix(matrixA.getRows(), matrixB.getColumns());
	
		if (matrixA.getColumns() == matrixB.getRows()) {  
			for (int j = 0; j < matrixA.getRows(); j++) {
				for (int k = 0; k < matrixB.getColumns(); k++) {
					double value = multiplyMatrix.getElement(j, k);
					for (int m = 0; m < matrixA.getColumns(); m++) {
						value += matrixA.getElement(j, m) * matrixB.getElement(m, k);
					}
					multiplyMatrix.assignElement(value, j, k);
				}
			}
		}	
		return multiplyMatrix;	
	}
	
	/**
	 * The difference method performs the subtraction operation on two m-by-n matrices; they must have the same number of
	 *      rows and columns. 
	 * @param matrixA     an m-by-n matrix
	 * @param matrixB     an m-by-n matrix
	 * @return differenceMatrix    an m-by-n matrix of corresponding elements equal to the difference of matrixB from matrixA  
	 */
	private Matrix difference(Matrix matrixA, Matrix matrixB) {
		
		Matrix differenceMatrix = new Matrix(matrixA.getRows(), matrixA.getColumns());
		
		if (matrixA.getRows() == matrixB.getRows() && matrixA.getColumns() == matrixB.getColumns()) {
			for (int j = 0; j < matrixA.getRows(); j++) {
				for (int k = 0; k < matrixA.getColumns(); k++) {
					differenceMatrix.assignElement(matrixA.getElement(j, k) - matrixB.getElement(j, k), j, k);
				}
			}
		}
		return differenceMatrix;
	}
	
	/**
	 * 
	 * @param covariance1
	 * @return
	 */
	private double calculateTrace(Matrix covariance1) {
		
		double trace = 0; 
		
		if (covariance1.getRows() == covariance1.getColumns()) {
			for (int i = 0; i < covariance1.getRows(); i++) {
				trace += covariance1.getElement(i, i);
			}
		}
		return trace;
	}
	
	/**
	 * The pivot method calculates the pivot value; the largest absolute value for a specified column i, of an m-by-n augmented matrix.
	 * @param i     integer value indicating column
	 * @param matrixA     an m-by-n augmented matrix
	 * @return pivot     integer value indicating the row that contains the highest absolute value for a specified column i
	 */
	private int pivot(int i, Matrix matrixA) {
		
		int pivot = i;
		
		for (int j = i + 1; j < matrixA.getRows(); j++) {	
			if (Math.abs(matrixA.getElement(j, i)) > Math.abs(matrixA.getElement(pivot, i))) {
				pivot = j;
			}
		}
		return pivot;
	}
	
	/**
	 * The rowSwap method swaps the elements in current row i with row pivot; which contains the pivot value, of an m-by-n augmented matrix. 
	 * @param i     integer value indicating current row
	 * @param pivot     integer value indicating pivot row 
	 * @param matrixA     an m-by-n augmented matrix
	 * @return matrixA     an m-by-n updated augmented matrix 
	 */
	private Matrix rowSwap(int i, int pivot, Matrix matrixA) {
		
		for (int col = 0; col < matrixA.getColumns(); col++) {
			double temp = matrixA.getElement(pivot, col);
			matrixA.assignElement(matrixA.getElement(i, col), pivot, col);
			matrixA.assignElement(temp, i, col);
		}
		return matrixA;
	}
	
	/**
	 * The rowDivision method divides the elements in row i by the first elements in row i, of an m-by-n augmented matrix. 
	 * @param i     integer value indicating row
	 * @param matrixA     an m-by-n augmented matrix     
	 * @return matrixA     an m-by-n updated augmented matrix
	 */
	private Matrix rowDivision(int i, Matrix matrixA) {
		
		double divisor = matrixA.getElement(i, i);
		
		for(int col = 0; col < matrixA.getColumns(); col++) {
			matrixA.assignElement(matrixA.getElement(i, col) / divisor, i, col);
		}
		return matrixA;
	}
	
	/**
	 * The clone method makes a copy of a Matrix
	 * @param originalM     an n-by-n matrix   
	 * @return cloneM     an n-by-n matrix clone
	 */
	private Matrix clone(Matrix originalM) {
		
		Matrix cloneM = new Matrix(originalM.getRows(), originalM.getColumns());
		
		for (int i = 0; i < originalM.getRows(); i++) {
			for (int j = 0; j < originalM.getColumns(); j++) {
				cloneM.assignElement(originalM.getElement(i, j), i, j);
			}
		}
		return cloneM;
	}
	
	/**
	 * The gaussReductionDeterminant method performs a sequence of operations on an m-by-n coefficient matrix, resulting in the determinant
	 *      of the matrix. 
	 * @param matrixA     an m-by-n coefficient matrix
	 * @return delta     a double value of the determinant 
	 */
	private double gaussReductionDeterminant(Matrix coefficient) {
		
		int r = 0;
		double delta;
		
		Matrix matrixA = clone(coefficient);
	
		for (int i = 0; i < matrixA.getRows(); i++) {
			 
			// Compute the pivot
			int pivot = pivot(i, matrixA);
			
			// Swap rows
			if (pivot > i) {
				matrixA = rowSwap(i, pivot, matrixA);
				r = r + 1;
			}	
			
			// Invalid matrix
			if(matrixA.getElement(i, i) == 0) {
				delta = 0;
				System.exit(0);
			}
			
			// Elimination step
			for (int j = 0; j < matrixA.getRows(); j++) {
				if (j > i) {
					double value = matrixA.getElement(j, i) / matrixA.getElement(i, i);
					for (int k = 0; k < matrixA.getColumns(); k++) {
						matrixA.assignElement(matrixA.getElement(j, k) - (value * matrixA.getElement(i, k)), j, k); 
					} 
				}
			}
		}
		
		// Calculate delta
		delta = Math.pow(-1, r) * matrixA.getElement(0, 0);
		for (int i = 1; i < matrixA.getRows(); i++) {
			delta *= matrixA.getElement(i, i);
		}
		return delta;
	}
	
	/**
	 * 
	 * @param matrixA
	 * @return trace
	 */
	private double trace(Matrix matrixA) {
		
		double trace = 0;
		
		if (matrixA.getRows() == matrixA.getColumns()) {
			for (int i = 0; i < matrixA.getRows(); i++) {
				trace += matrixA.getElement(i, i);
			}
		}
		else {
			System.out.println("Matrix must be a square.");
		}
		
		return trace;
	}
	
	/**
	 * The identityMatrix method generates an n-by-n matrix in which all elements of the principal diagonal are ones,
	 *      and all other elements are zeros. 
	 * @param n     integer value indicating number of row and columns
	 * @return identity     an n-by-n identity matrix
	 */
	private Matrix identityMatrix(int n) {
		Matrix identity = new Matrix(n, n);
		for (int i = 0; i < n; i++) {
			identity.assignElement(1, i, i);
		}
		return identity;
	}
	

	/**
	 * The augment method creates an augmented m-by-(n+1) matrix, by appending the columns of an m-by-n matrix and m-by-1 matrix. 
	 * @param matrixA     an m-by-n matrix
	 * @param matrixB     an m-by-1 matrix
	 * @return augmentedMatrix     an m-by-(n+1) matrix by appending the columns of matrixB to matrixA 
	 */
	private Matrix augment(Matrix matrixA, Matrix matrixB) {
		
		Matrix augmentedMatrix = new Matrix(matrixA.getRows(), matrixA.getColumns() + matrixB.getColumns());
		
		// "A" portion of coefficient matrix
		for (int i = 0; i < matrixA.getRows(); i++) {
			for (int j = 0; j < matrixA.getColumns(); j++) {
				augmentedMatrix.assignElement(matrixA.getElement(i, j), i, j);
			}
		}
		
		// "B" portion of coefficient matrix
		for (int k = 0; k < matrixB.getRows(); k++) {
			for (int l = 0; l < matrixB.getColumns(); l++) {
				augmentedMatrix.assignElement(matrixB.getElement(k, l), k, matrixA.getRows() + l);
			}
		}
		return augmentedMatrix;
	}
	
	/**
	 * The gaussJordanElimination method transforms an m-by-n augmented matrix into reduced row-echelon form, by a series of row operations.
	 * @param matrixA     an m-by-n matrix 
	 * @param matrixB     an m-by-1 matrix
	 * @return augment     and m-by-(n+1) matrix in reduced echelon form
	 */
	private Matrix gaussJordanElimination(Matrix matrixA, Matrix matrixB) {
		
		int error = 1;
		Matrix augment = augment(matrixA, matrixB);
		
		for (int i = 0; i < matrixA.getRows(); i++) {
			 
			// Compute the pivot
			int pivot = pivot(i, augment);
			
			// Swap rows
			if (pivot > i) {
				augment = rowSwap(i, pivot, augment);
			}	
			
			// Invalid matrix
			if(augment.getElement(i, i) == 0) {
				error = 0;
				return augment;
				//System.exit(0);
			}
			
			// Divide row
			augment = rowDivision(i, augment);
			
			// Elimination step
			for (int j = 0; j < matrixA.getRows(); j++) {
				if (i != j) {
					double value = augment.getElement(j, i);
					for (int k = 0; k < augment.getColumns(); k++) {
						double newValue = augment.getElement(j, k) - (value * augment.getElement(i, k));
						augment.assignElement(newValue, j, k); 
					} 
				}
			}
			// Zero out small values
			for (int m = 0; m < augment.getRows(); m++) {
				for (int n = 0; n < augment.getColumns(); n++) {
					if (Math.abs(augment.getElement(m, n)) < 0.00001) {
						augment.assignElement(0, m, n);
					}
				}
			}
		}
		return augment;
	}
	
	
	
	
	//--------------------------------------------------------------------------------------------------------
	
	private City[] exhaustiveTour; City[] randomTour; City[] geneticTour; City[] simmulatedTour;
	private int numberOfCities = 0;
	private double numberOfPermutations = 0;
	private double sumOfPermutations = 0;
	private double squaredSum = 0;
	private double shortestPath = 999;
	private double longestPath = 0;
	private City[] shortestTrip;
	private City[] longestTrip;
	private double[][] exhaustiveBin = new double[1][100]; double[][] randomBin = new double[1][99];
	private double[][] geneticBin = new double[1][99]; double[][] simmulatedBin = new double[1][99];
	private double exhaustiveMean; double randomMean; double geneticMean; double simmulatedMean; 
	private double exhaustiveSD; double randomSD; double geneticSD; double simmulatedSD; 
	private static double globalLong = 4.06;
	private static double globalShort = 2.57;
	
	private void travelingSalesmanProblem() {
		
		readFileTSP();
		
		City[] practice = new City[6];
		for (int i = 0; i < 6; i++) {
			practice[i] = exhaustiveTour[i];
		}
		
		exhaustiveSearch(practice);
		//exhaustiveSearch(exhaustiveTour);
		reset();
		//randomSearch(randomTour);
		//reset();
		//geneticSearch(geneticTour);
		//reset();
		//simmulatedSearch(simmulatedTour);
		//reset();
		
	}
	
	private void exhaustiveSearch(City[] practice) {
		
		//permutation(exhaustiveTour, exhaustive.length);
		permutation(practice, practice.length);
		
		this.exhaustiveMean = sumOfPermutations / numberOfPermutations;
		System.out.println("The Mean for Exhaustive Search is: " + exhaustiveMean);
		System.out.println("The Longest Path for Exhaustive Search is: " + longestPath);
		System.out.println("The Shortest Path for Exhaustive Search is: " + shortestPath + "\n");
		
		String exhaustiveLong = new String(); String exhaustiveShort = new String();
		for (int i = 0; i < practice.length; i++) {
			exhaustiveLong += longestTrip[i].getLabel();
			exhaustiveShort += shortestTrip[i].getLabel();
		}
		
		System.out.println("The Trip Order for Longest Path is: " + exhaustiveLong + longestTrip[0].getLabel());
		System.out.println("The Trip Order for Shortest Path is: " + exhaustiveShort + shortestTrip[0].getLabel());
		
		this.exhaustiveSD = standardDeviation();
		System.out.println("The Standard Deviation for Exhaustive Search is: " + exhaustiveSD);
		
		System.out.println("Exhaustive Bin");
		System.out.println(exhaustiveBin.length);
		for (int ii = 0; ii < exhaustiveBin[0].length; ii++) {
			System.out.println(exhaustiveBin[0][ii]);
		}
		return;
	}
	
    /**
     * 
     * @param tour
     * @param n
     */
    private void permutation(City[] eTour, int n) {
    	
        if (n == 1) {
        		// Remove for final project
        		System.out.println("Permutation" + " " + (int)(numberOfPermutations + 1));
        		for (int i = 0; i < eTour.length; i++) {
        			System.out.println(eTour[i]);
        		}
        		System.out.println(eTour[0]);
        		
        		double exhaustiveTD = tripDistance(eTour);
        		updateData(exhaustiveTD, eTour);
        		updateBin(exhaustiveBin, exhaustiveTD);
        		numberOfPermutations++;
        		return;
        }
        
        for (int i = 0; i < n; i++) {
            swap(eTour, i, n - 1);
            permutation(eTour, n - 1);
            swap(eTour, i, n - 1);
        }
    }
	
    /**
     * 
     * @param randomTour
     */
	private void randomSearch(City[] randomTour) {
		
		random(randomTour);
		
		this.randomMean = sumOfPermutations / numberOfPermutations;
		System.out.println("The Mean for Random Search is: " + randomMean);
		System.out.println("The Longest Path for Random Search is: " + longestPath);
		System.out.println("The Shortest Path for Random Search is: " + shortestPath + "\n");
		
		String randomLong = new String(); String randomShort = new String();
		for (int i = 0; i < randomTour.length; i++) {
			randomLong += longestTrip[i].getLabel();
			randomShort += shortestTrip[i].getLabel();
		}
		System.out.println("The Trip Order for Longest Path is: " + randomLong + longestTrip[0].getLabel());
		System.out.println("The Trip Order for Shortest Path is: " + randomShort + shortestTrip[0].getLabel());
		
		this.randomSD = standardDeviation();
		System.out.println("The Standard Deviation for Random Search is: " + randomSD);
		
		System.out.println("Random Bin");
		for (int j = 0; j < randomBin[0].length; j++) {
			System.out.println(randomBin[0][j]);
		}
		return;
	}
	
	/**
	 * 
	 * @param randomTour
	 */
	private void random(City[] rTour) {
		
		while (numberOfPermutations < 24) {
			
			for (int i = 0; i < 5; i++) {
				Random rand = new Random();
				int value = rand.nextInt(rTour.length);
				swap(rTour, i, value);
			}
			// Remove for final project
			System.out.println("Permutation" + " " + (int)(numberOfPermutations + 1));
			for (int i = 0; i < rTour.length; i++) {
				System.out.println(rTour[i]);
			}
			System.out.println(rTour[0]);
			
			double randomTD = tripDistance(rTour);
			updateData(randomTD, rTour);
			updateBin(randomBin, randomTD);
			numberOfPermutations++;
		}
		return;
	}
	
	/**
	 * 
	 * @param geneticTour
	 */
	private void geneticSearch(City[] geneticTour) {
		
		ArrayList<City[]> geneticPopulation = new ArrayList();
		
		// Generate an initial population of 50 random individuals
		for (int i = 0; i < 50; i++) {
			City[] random = generateRandomIndividual(geneticTour);
			geneticPopulation.add(random);
		}
		
		for (int j = 0; j < 50; j++) {
			ArrayList<City[]> temp = new ArrayList();
			temp = genetic(geneticTour, geneticPopulation);
			geneticPopulation.clear();
			geneticPopulation = temp;	
		}
		
		this.geneticMean = sumOfPermutations / numberOfPermutations;
		System.out.println("The Mean for Genetic Search is: " + geneticMean);
		System.out.println("The Longest Path for Genetic Search is: " + longestPath);
		System.out.println("The Shortest Path for Genetic Search is: " + shortestPath + "\n");
		
		String geneticLong = new String(); String geneticShort = new String();
		for (int i = 0; i < geneticTour.length; i++) {
			geneticLong += longestTrip[i].getLabel();
			geneticShort += shortestTrip[i].getLabel();
		}
		System.out.println("The Trip Order for Longest Path is: " + geneticLong + longestTrip[0].getLabel());
		System.out.println("The Trip Order for Shortest Path is: " + geneticShort + shortestTrip[0].getLabel());
		
		this.geneticSD = standardDeviation();
		System.out.println("The Standard Deviation for Genetic Search is: " + geneticSD);
		
		System.out.println("Genetic Bin");
		for (int j = 0; j < geneticBin[0].length; j++) {
			System.out.println(geneticBin[0][j]);
		}
		
		return;
	}
	
	
	/**
	 * 
	 * @param geneticTour
	 */
	private ArrayList<City[]> genetic(City[] gTour, ArrayList<City[]> population) {
		
		ArrayList<City[]> newPopulation = new ArrayList();
		ArrayList<City[]> finalPopulation = new ArrayList();
		City[] originalFittest = getFittest(population);
		
		double geneticTD = tripDistance(originalFittest);
		updateData(geneticTD, originalFittest);
		updateBin(geneticBin, geneticTD);
		numberOfPermutations++;
		
		newPopulation.add(originalFittest);
		
		for (int j = 1; j < population.size(); j++) {
			// Select parents 
			City[] parentA = tournamentSelection(gTour);
			City[] parentB = tournamentSelection(gTour);
			// Crossover parents
			City[] crossover = crossover(parentA, parentB);
			// Add child to new population
			newPopulation.add(crossover);
		}
		
		// Mutate the new population a bit to add some new genetic material
		for (int k = 0; k < newPopulation.size(); k++) {
			City[] mutation = swapMutation(newPopulation.get(k));
			finalPopulation.add(mutation);
		}
		return finalPopulation;	
	}
	
	/**
	 * 
	 * @param tour
	 * @return
	 */
	private City[] swapMutation(City[] tour) {
		
		City[] mutatedTour = tour.clone();
		double mutationRate = 0.15;
		Random rand = new Random();
		
		for (int i = 0; i < tour.length; i++) {
			if (Math.random() < mutationRate) {
				int index = rand.nextInt(tour.length);
				swap(mutatedTour, i, index);
			}
		}
		return mutatedTour;
	}
	
	/**
	 * 
	 * @param parentA
	 * @param parentB
	 * @return
	 */
	private City[] crossover(City[] parentA, City[] parentB) {
		
		// Create new child tour
		City[] child = new City[parentA.length];
		
		// Randomly generate start and end indices 
		Random rand = new Random();
		int startIndex = rand.nextInt(parentA.length);
		int endIndex = rand.nextInt(parentA.length);
		
		// Adjust startIndex to reflect array position in java
		if (startIndex != 0) {
			startIndex = startIndex - 1;
		}
		// Adjust endIndex to reflect array positioning in java
		if (endIndex != 0) {
			endIndex = endIndex - 1;
		}
		
		// Sub tour from parent A
		if (startIndex < endIndex) {
			for (int i = startIndex; i <= endIndex; i++) {
				child[i] = parentA[i];
			}
		}
		else if (startIndex > endIndex) {
			for (int j = startIndex; j < parentA.length; j++) {
				child[j] = parentA[j];
			}
			for (int k = 0; k <= endIndex; k++) {
				child[k] = parentA[k];
			}
		}
		
		int index = 0;
		
		// Sub tour from Parent B
		for (int i = 0; i < parentB.length; i++) {
			boolean contains = false;
			
			// Verify parentB City is not already found in child 
			for (int j = 0; j < child.length; j++) {
				if (parentB[i] == child[j]) {
					contains = true;
				}
			}
			// If parentB City is not found in child, find first available spot in child
			// and set equal to parentB City. 
			if (contains == false) {
				for (int k = index; k < child.length; k++) {
					if (child[k] == null) {
						child[k] = parentB[i];
						index = k;
						break;
					}
				}
			}
		}
		return child;
	}
	
	/**
	 * 
	 * @param tour
	 * @return
	 */
	private City[] generateRandomIndividual(City[] tour) {
		
		City[] cloneTour = tour.clone();
		City[] randomParent = new City[tour.length];
		int counter = 0; 
		
		// Randomly generate randomParent
		for (int i = cloneTour.length - 1; i > 0; i--) {
			Random rand = new Random();
			int value = rand.nextInt(i + 1);
			randomParent[counter] = cloneTour[value];
			swap(cloneTour, i, value);
			counter++;
		}
		
		// Add remaining City to randomParent
		randomParent[tour.length - 1] = cloneTour[0];		
		
		// Verify randomTour contains no duplicate cities. 
		/**
		System.out.println("Permutation" + " " + (int)(numberOfPermutations + 1));
		for (int i = 0; i < randomParent.length; i++) {
			System.out.println(randomParent[i]);
		}
		
		for (int i = 0; i < randomParent.length; i++) {
			boolean contains = false;
			for (int j = 0; i < cloneTour.length; j++) {
				if (randomParent[i].getLabel().equals(cloneTour[j].getLabel())) {
					contains = true;
					break;
				}
			}
			if (contains != true) {
				System.out.println("Error!");
			}
		}
		System.out.println("Success!");
		**/
		return randomParent;	
	}
	
	/**
	 * 
	 * @param gTour
	 * @return
	 */
	private City[] tournamentSelection(City[] gTour) {
		
		// Create tournament population
		ArrayList<City[]> tournament = new ArrayList();
		
		for (int i = 0; i < 10; i++) {
			City[] random = generateRandomIndividual(gTour);
			tournament.add(random);
		}
		City[] fittest = getFittest(tournament);
		return fittest;
	}
	
	/**
	 * 
	 * @param population
	 * @return
	 */
	private City[] getFittest(ArrayList<City[]> population) {
		
		double fitness = fitness(population.get(0)); 
		City[] fittest = population.get(0); 
		
		for (int i = 1; i < population.size(); i++) {
			double tempFitness = fitness(population.get(i));
			if (tempFitness > fitness) {
				fitness = tempFitness;
				fittest = population.get(i);
			}
		}
		return fittest;
	}
	
	/**
	 * 
	 * @param tour
	 * @return
	 */
	private double fitness(City[] tour) {
		
		double fitness = 1 / tripDistance(tour);
		return fitness;
	}
	
	/**
	 * 
	 * @param simmulatedTour
	 */
	private void simmulatedSearch(City[] simmulatedTour) {
		
		double startTemp = 5;
		int numIterations = 50;
		double coolingRate = 0.99;
		
		
		simmulated(simmulatedTour, startTemp, numIterations, coolingRate);
		
		this.simmulatedMean = sumOfPermutations / numberOfPermutations;
		System.out.println("The Mean for Simmulated Annealing is: " + simmulatedMean);
		System.out.println("The Longest Path for Simmulated Annealing is: " + longestPath);
		System.out.println("The Shortest Path for Simmulated Annealing is: " + shortestPath + "\n");
		
		String simmulatedLong = new String(); String simmulatedShort = new String();
		for (int i = 0; i < simmulatedTour.length; i++) {
			simmulatedLong += longestTrip[i].getLabel();
			simmulatedShort += shortestTrip[i].getLabel();
		}
		System.out.println("The Trip Order for Longest Path is: " + simmulatedLong + longestTrip[0].getLabel());
		System.out.println("The Trip Order for Shortest Path is: " + simmulatedShort + shortestTrip[0].getLabel());
		
		this.simmulatedSD = standardDeviation();
		System.out.println("The Standard Deviation for Simmulated Annealing is: " + simmulatedSD);
		
		System.out.println("Simmulated Bin");
		for (int j = 0; j < simmulatedBin[0].length; j++) {
			System.out.println(simmulatedBin[0][j]);
		}
		return;
	}
	
	/**
	 * 
	 */
	private void simmulated(City[] sTour, double startTemp, int iterations, double coolingRate) {
		
		double t = startTemp;
		City[] startSolution = generateRandomIndividual(sTour);
		double bestDistance = tripDistance(startSolution);
		City[] currentSolution = startSolution;
		
		for (int i = 0; i < iterations; i++) {
			if (t > 0.1) {
				Random rand = new Random();
				int j = rand.nextInt(sTour.length);
				int k = rand.nextInt(sTour.length);
				swap(currentSolution, j, k);
				double currentDistance = tripDistance(currentSolution);
				if(currentDistance < bestDistance) {
					bestDistance = currentDistance;
				}
				else if (Math.exp((bestDistance - currentDistance) / t) < Math.random()) {
					swap(currentSolution, j, k);
				}
				t *= coolingRate;
				
				updateData(currentDistance, currentSolution);
				updateBin(simmulatedBin, currentDistance);
				numberOfPermutations++;
        		}
        			
			else {
				return;
			}
		}
	}
	
	/**
	 * 
	 */
	private void readFileTSP() {
		
		// Read in data from text file
		ArrayList<String> temp = new ArrayList<>();
		Scanner input = null;
		File infile = new File("Resource/tsp.txt");
				
		try {
			input = new Scanner(infile, "UTF-8");
		} catch (FileNotFoundException e) {
			System.out.println("File not found.");
			e.printStackTrace();
		}
		while (input.hasNextLine()) {
			temp.add(input.nextLine());
			numberOfCities++;
		}	
		input.close();
		System.out.println(numberOfCities);
		exhaustiveTour = new City[numberOfCities];
		int lineCounterTSP = 0;
				
		for (String ln: temp) {
			
			@SuppressWarnings("resource")
			Scanner lineScannerTSP = new Scanner(ln);
      		lineScannerTSP.useDelimiter(";");
		        			
		     // Create City 
		     double xPos = Double.parseDouble(lineScannerTSP.next().trim());
		     double yPos = Double.parseDouble(lineScannerTSP.next().trim());
		     String label = lineScannerTSP.next().trim();
		     City point = new City(label, xPos, yPos);
		     exhaustiveTour[lineCounterTSP] = point;	
		        
		     lineCounterTSP += 1;
		}	
		randomTour = exhaustiveTour.clone();
		geneticTour = exhaustiveTour.clone();
		simmulatedTour = exhaustiveTour.clone();
	}
	
	/**
	 * swap the City(points) at indices i and j
	 * @param tour
	 * @param i
	 * @param j
	 */
    private void swap(City[] tour, int i, int j) {
        City city = tour[i];
        tour[i] = tour[j];
        tour[j] = city;
    }
	
	/**
	 * 
	 * @param pointX
	 * @param pointY
	 * @return
	 */
	private double distance(City aC, City bC) {
		
		double x = aC.getX(); double a = bC.getX();
		double y = aC.getY(); double b = bC.getY();
		
		double dist = Math.sqrt(Math.pow((x - a), 2) + Math.pow((y - b), 2));
		
		return dist;
	}
	
	/**
	 * 
	 * @param tour
	 * @return
	 */
	private double tripDistance(City[] tour) {
		
		double dist = 0; 
		
		for (int i = 0; i < tour.length - 1; i++) {
			dist += distance(tour[i], tour[i+1]);
		}
		dist += distance(tour[tour.length - 1], tour[0]);
		
		return dist;	
	}
	
	/**
	 * 
	 * @param tourDist
	 * @param tour
	 */
	private void updateData(double tourDist, City[] tour) {
		
		sumOfPermutations += tourDist;
		squaredSum += Math.pow(tourDist, 2);
		
		if (tourDist < shortestPath) {
			shortestPath = tourDist;
			shortestTrip = tour.clone();
		}
		if (tourDist > longestPath) {
			longestPath = tourDist;
			longestTrip = tour.clone();
		}
		return;
	}
	
	/**
	 * 
	 * @return
	 */
	private double standardDeviation() {
		
		double root = (squaredSum - (Math.pow(sumOfPermutations, 2) / (double)numberOfPermutations));
		double stanDev = Math.sqrt(root/(double)(numberOfPermutations - 1));
		return stanDev;
	}
	
	/**
	 * 
	 * @param bin
	 * @param tripDist
	 * @return
	 */
	private double[][] updateBin(double[][] bin, double tripDist) {
		
		double value = (tripDist - globalShort) / (globalLong - globalShort);
		int binNumber = (int) (value * 100);
		bin[0][binNumber] = bin[0][binNumber] + 1;
		
		return bin;
	}
	
	/**
	 * 
	 */
	private void reset() {
		
		sumOfPermutations = 0;
		numberOfPermutations = 0;
		this.squaredSum = 0;
		this.shortestPath = 9999;
		this.longestPath = 0;
		
		return;
	}
	
	//----------------------------------------------------------------------------------------------------


	
	/**
	 * Performs Leverrier's method to identify coefficients in polynomial function.
	 * @param matrix A
	 * @return String equation
	 */	
	private void leverrier(Matrix matrixA) {
		
		if (matrixA.getRows() == matrixA.getColumns()) {
			Matrix matrixB = clone(matrixA);
			double aN = -trace(matrixB);
			
			System.out.print("1.0x^"+ (matrixA.getRows() + " + " + aN + "x^" + (matrixA.getRows()-1)));
		
			for (int k = matrixB.getRows() - 1; k >= 1; k--) {
			 
				Matrix I = matrixScalar(identityMatrix(matrixB.getRows()), aN);
				matrixB = multiply(matrixA, add(matrixB, I));
				aN = -(trace(matrixB) / (matrixA.getRows() - k + 1));
				
				if (k != 1) {
					if (aN > 0) {
						System.out.print(" + " + aN + "x^" + (k - 1));
					}
					else {
						System.out.print(" - " + Math.abs(aN) + "x^" + (k - 1));
					}
				}
				
				else {
					if (aN > 0) {
						System.out.println(" + " + aN);
					}
					else {
						System.out.println(" - " + Math.abs(aN));
					}
				}
			}
		}
	}
	
	/**
	 * 
	 * @param matrixA
	 * @return
	 */
	private Matrix householder(Matrix matrixA) {

		int n = matrixA.getRows();
		Matrix B = clone(matrixA);
		
		for (int k = 0; k < n - 2; k++) {
			
			double alpha; 
			double sum = 0;
			Matrix vector = new Matrix(n - 1 - k, 1);
			Matrix u = new Matrix(B.getRows() - k, 1);
			Matrix Q = identityMatrix(B.getRows() - k);
			Matrix A = new Matrix(B.getRows() - k, B.getColumns() - k);
			
			// Assign appropriate values to Matrix A from Matrix B
			for (int i = 0 + k; i < n; i++) {
				for (int j = 0 + k; j < n; j++) {
					A.assignElement(B.getElement(i, j), i - k, j - k);
				}
			}
			
			// Assign values to vector 
			for (int i = k + 1; i < n; i++) {
				vector.assignElement(B.getElement(i, k), i - k - 1, 0);
			}
			
			// Calculate sum of squared vector values
			for (int i = 0; i < vector.getRows(); i++) {
				sum += Math.pow(vector.getElement(i, 0), 2);
			}
			
			// Determine if alpha is positive or negative
			if (vector.getElement(0, 0) >= 0) {
				alpha = Math.sqrt(sum);
			}
			else {
				alpha = -Math.sqrt(sum);
			}
			
			// Assign values to Matrix u
			u.assignElement(0, 0, 0);
			u.assignElement(alpha + vector.getElement(0, 0), 1, 0);
			for (int i = 2; i < B.getRows() - k; i++) {
				u.assignElement(vector.getElement(i - 1, 0), i, 0);
			}
			
			// Transpose Matrix u
			Matrix uT = transpose(u);
			
			// Multiply Matrix u by Matrix uT
			Matrix uuT = multiply(u, uT);
			
			// Multiply Matrix uT by Matrix u
			Matrix uTu = multiply(uT, u);
			
			// Calculate scalar
			double scalar = 2 / (uTu.getElement(0, 0));
			
			// Scale Matrix uuT by scalar
			Matrix uuTS = matrixScalar(uuT, (scalar));
			
			// Difference of Matrix Q (Identity Matrix) and Matrix uuTS; 
			Matrix P = difference(Q, uuTS);
			
			// Q = Multiply Matrix Q (Identity Matrix 
			Q = multiply(Q, P);
			
			// Calculate Matrix A by multiplying Matrix P by Matrix A by Matrix P
			Matrix PA = multiply(P, A);
			Matrix PAP = multiply(PA, P);
			
			// Reassign a value of 0 to very small numbers of Matrix PAP
			for (int i = 0; i < PAP.getRows(); i++) {
				for (int j = 0; j < PAP.getColumns(); j++) {
					if (Math.abs(PAP.getElement(i, j)) < 0.00001) {
						PAP.assignElement(0, i, j);
					}
				}
			}
			
			// Assign calculated values to Matrix B
			for (int i = 0; i < PAP.getRows(); i++) {
				for (int j = 0; j < PAP.getColumns(); j++) {
						B.assignElement(PAP.getElement(i, j), i + k, j + k);
				}
			}
			System.out.println(B);
		}
		// Matrix B in upper-Hessenberg form
		return B;
	}
	
	/**
	 * 
	 * @param matrixA
	 */
	private void getEigenValues(Matrix matrixA) {
		
		double traceA = trace(matrixA);
		double determinantA = gaussReductionDeterminant(matrixA);
		
		double lambdaPos = (traceA + Math.sqrt(Math.pow(traceA, 2) - (4*determinantA))) / 2;
		double lambdaNeg = (traceA - Math.sqrt(Math.pow(traceA, 2) - (4*determinantA))) / 2;
		
		System.out.println("The Eigenvalues of the Covariance Matrix are: " + lambdaPos + ", " + lambdaNeg);
		eigenvector(matrixA, lambdaPos);
		eigenvector(matrixA, lambdaNeg);
	}
	
	/**
	 * 
	 * @param matrixA
	 * @param lambda
	 */
	private void eigenvector(Matrix matrixA, double lambda) {
		
		Matrix identity = identityMatrix(matrixA.getRows());
		Matrix lambdaIdentity = matrixScalar(identity, lambda); 
		Matrix eV = difference(matrixA, lambdaIdentity);
		
		Matrix xY = new Matrix(matrixA.getRows(), 1);
		xY.assignElement(0.0, 0, 0);
		xY.assignElement(0.0, 1, 0);
		System.out.println(eV);
		Matrix eigV = gaussJordanElimination(eV, xY);
		System.out.println("EigV" + eigV);
	}
	
	

	/**
	 * Performs power method to estimate largest eigenvalue for the inputed matrix
	 * @param matrix A : the matrix for which the eigenvalue is being estimated.
	 * @return double eigenvalue
	 */	
	private double powerMethod(Matrix matrixA) {
		
		//Matrix matrixAA = clone(matrixA);
		double epsilon = 1.0E-15;
		int m = 5;
		int k = 0;
		double[][] test = new double[matrixA.getRows()][1];
		double xSum = 0;
		double xNorm = 0;
		double rSum = 0;
		double rNorm = 0;
		double u = 0;
		
		// Populate arbitrary vector
		for (int i = 0; i < matrixA.getRows(); i++) {
			test[i][0] = 1;
		}	
		
		// Generate Matrix Y
		Matrix matrixY = new Matrix(test);
		
		// Multiply Matrix A by Matrix Y
		Matrix matrixX = multiply(matrixA, matrixY);
		System.out.println(matrixX);
		do {
			
			// Compute ||x|| norm for Matrix X (vector)
			for (int i = 0; i < matrixX.getRows(); i++) {
				xSum += Math.pow(matrixX.getElement(i, 0), 2);
			}
			xNorm = Math.sqrt(xSum);
			
			// Normalize x!
			for (int i = 0; i < matrixX.getRows(); i++) {
				matrixX.assignElement((matrixA.getElement(i, 0) / xNorm), i, 0);	
			}	
			
			// Set Matrix Y = Matrix X
			matrixY = matrixX; 
			System.out.println(matrixY);
			// Multiply Matrix A by Matrix Y
			matrixX = multiply(matrixA, matrixY);
			System.out.println(matrixX);
			// Compute transpose of Matrix Y
			Matrix yT = transpose(matrixY);
			
			// Compute numerator for dominant eigenvalue 
			Matrix uNumerator = multiply(yT, matrixX);
			
			//Compute denominator for dominant eigenvalue 
			Matrix uDenominator = multiply(yT, matrixY);
			
			// Compute dominant eigenvalue
			u = ((uNumerator.getElement(0, 0)) / (uDenominator.getElement(0, 0)));
			
			// Compute u * matrixY
			Matrix uY = matrixScalar(matrixY, u);
			
			// Subtract Matrix X from Matrix uY
            Matrix r = difference(uY, matrixX);
            
            // Compute ||r|| norm for Matrix r (vector)
            for (int i = 0; i < r.getRows(); i++) {
        			rSum += Math.pow(r.getElement(i, 0), 2);	
        		}
        		rNorm = Math.sqrt(rSum);
        		
        		// Increment counter
        		k++;  
		}

	while ((rNorm > epsilon) && (k < m));
		System.out.println(matrixY);
		return u;
	}
}


