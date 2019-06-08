import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;
/*
 * Jonathan Neff
 * Project_1
 * 10/16/18
 */

public class MatrixProject {
	
	// Class Properties
	ArrayList<Matrix> matrixClass1 = new ArrayList<>();
	ArrayList<Matrix> matrixClass2 = new ArrayList<>();
	
	ArrayList<double[][]> discClass1 = new ArrayList<>();
	ArrayList<double[][]> discClass2 = new ArrayList<>();
	
	Matrix meanClass1;
	Matrix meanClass2;
	
	Matrix covarianceClass1;
	Matrix covarianceClass2;
	
	Matrix gaussJordan1;
	Matrix gaussJordan2;
	
	double determinant1;
	double determinant2;
	
	Matrix inverseClass1;
	Matrix inverseClass2;
	
	ArrayList<Matrix> boundaryPoints;
	
	double pWI = 0; 
	
	Matrix gaussJordanC;
	double determinantC;
	Matrix inverseC;
	double determinantInverseC;
	double conditionNumberC;
	
	
	public static void main(String[] args) {
		MatrixProject mp = new MatrixProject();
	}
	
	
	public MatrixProject() {
		
		// Read in data from text file
		ArrayList<String> temp = new ArrayList<>();
		Scanner input = null;
		File infile = new File("Resource/test.txt");
		
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
			if (lineCounter >= 2) {
				
				@SuppressWarnings("resource")
				Scanner lineScanner = new Scanner(ln);
        			lineScanner.useDelimiter(";");
        			
        			// Create Class 1 2x1 matrix and add to ArrayList
        			double[][] vectorA = new double[2][1];
        			vectorA[0][0] = Double.parseDouble(lineScanner.next().trim());
        			vectorA[1][0] = Double.parseDouble(lineScanner.next().trim());
        			Matrix matrixA = new Matrix(vectorA);
        			this.matrixClass1.add(matrixA);
        			
        			// Create Class 2 2x1 matrix and add to ArrayList
        			double[][] vectorB = new double[2][1];
        			vectorB[0][0] = Double.parseDouble(lineScanner.next().trim());
        			vectorB[1][0] = Double.parseDouble(lineScanner.next().trim());
        			Matrix matrixB = new Matrix(vectorB);
        			this.matrixClass2.add(matrixB);
        			
        			lineCounter += 1;
        			pWI += 1;
			}
			
			lineCounter += 1;
		}
		
		// Problem 1: Compute the mean for each Class:
		this.meanClass1 = computeMean(matrixClass1);
		this.meanClass2 = computeMean(matrixClass2);
		
		System.out.println("The Mean for Class 1 is: " + "x = " + meanClass1.getElement(0, 0) + "     " + "y = " + meanClass1.getElement(1, 0));
		System.out.println("The Mean for Class 2 is: " + "x = " + meanClass2.getElement(0, 0) + "     " + "y = " + meanClass2.getElement(1, 0) + "\n");
		
		// Problem 2: Compute the covariance matrix for each Class:
		this.covarianceClass1 = computeCovariance(matrixClass1, meanClass1);
		this.covarianceClass2 = computeCovariance(matrixClass2, meanClass2);
		
		System.out.println("The Covariance Matrix for Class 1 is: " + "\n" + covarianceClass1.getElement(0, 0) + "   " + covarianceClass1.getElement(0, 1) + "\n" + covarianceClass1.getElement(1, 0) + "   " + covarianceClass1.getElement(1, 1) + "\n");
		System.out.println("The Covariance Matrix for Class 2 is: " + "\n" + covarianceClass2.getElement(0, 0) + "   " + covarianceClass2.getElement(0, 1) + "\n" + covarianceClass2.getElement(1, 0) + "   " + covarianceClass2.getElement(1, 1) + "\n");
		
		// Problem 3: Compute the determinant for each Class:
		this.determinant1 = gaussReductionDeterminant(covarianceClass1);
		this.determinant2 = gaussReductionDeterminant(covarianceClass2);
		
		System.out.println("The Determinant for Class 1 is :" + " " + determinant1);
		System.out.println("The Determinant for Class 2 is :" + " " + determinant2 + "\n");
		
		// Problem 4: Compute the inverse matrix for each Class:
		this.inverseClass1 = matrixInverse(covarianceClass1);
		this.inverseClass2 = matrixInverse(covarianceClass2);
				
		System.out.println("The Inverse Matrix for Class 1 is: " + "\n" + inverseClass1.getElement(0, 0) + "   " + inverseClass1.getElement(0, 1) + "\n" + inverseClass1.getElement(1, 0) + "   " + inverseClass1.getElement(1, 1) + "\n");
		System.out.println("The Inverse Matrix for Class 2 is: " + "\n" + inverseClass2.getElement(0, 0) + "   " + inverseClass2.getElement(0, 1) + "\n" + inverseClass2.getElement(1, 0) + "   " + inverseClass2.getElement(1, 1) + "\n");
		
		// Problem 6: Compute the discriminant of the mean for each Class:
		double[][] meanDisc1 = calculateDiscriminant(meanClass1);
		double[][] meanDisc2 = calculateDiscriminant(meanClass2);
		
		System.out.println("Class 1 Mean Discriminant: ");
		System.out.println("[" + meanDisc1[0][0] + ", " + meanDisc1[0][1] + ", " + meanDisc1[0][2] + ", " + meanDisc1[0][3] + "]" + "\n");
		System.out.println("Class 2 Mean Discriminant: ");
		System.out.println("[" + meanDisc2[0][0] + ", " + meanDisc2[0][1] + ", " + meanDisc2[0][2] + ", " + meanDisc2[0][3] + "]" + "\n");
		
		// Problem 7a:
		// Compute the discriminant for every point in each Class:
		for (int i = 0; i < matrixClass1.size(); i++) {
			double[][] tempA = calculateDiscriminant(matrixClass1.get(i)); 
			double[][] tempB = calculateDiscriminant(matrixClass2.get(i));
			
			if (tempA[0][3] > tempA[0][2]) {
				discClass1.add(tempA);
			}
			if (tempB[0][2] > tempB[0][3]) {
				discClass2.add(tempB);
			}
		}
		
		// Class 1 misclassified points:
		System.out.println("The number of misscalssified points for Class 1 is: " + discClass1.size());
		System.out.println("x" + "             " + "y" + "            " + "g1(x)" + "               " + "g2(x)");
		for (int i = 0; i < discClass1.size(); i++) {
			double[][] tempo = discClass1.get(i);
			System.out.println("[" + tempo[0][0] + ", " + tempo[0][1] + ", " + tempo[0][2] + ", " + tempo[0][3] + "]");
		}
		System.out.println();
		
		// Class 2 misclassified points:
		System.out.println("The number of missclassified points for Class 2 is: " + discClass2.size());
		System.out.println("x" + "              " + "y" + "             " + "g1(x)" + "                " + "g2(x)");
		for (int i = 0; i < discClass2.size(); i++) {
			double[][] tempo = discClass2.get(i);
			System.out.println("[" + tempo[0][0] + ", " + tempo[0][1] + ", " + tempo[0][2] + ", " + tempo[0][3] + "]");
		}
		System.out.println();
		
		// Problem 7b:
		System.out.println("The number of correctly identified points for Class 1 is: " + (int)(pWI - discClass1.size()));
		System.out.println("The number of correctly identified points for Class 2 is: " + (int)(pWI - discClass2.size()) + "\n");
		
		
		// Problem 8: Create boundary
		boundaryPoints = createBoundary(matrixClass1, matrixClass2);
		
		// Problem 9: Linear systems
		double[][] partA = {
				{0.0, 1.0, 3.0, -1.0, 1.0, 0.0, -1.0, -1.0},
				{5.0, 0.0, 2.0, 0.0, -1.0, 3.0, 1.0, 1.0},
				{2.0, -2.0, 2.0, -1.0, -1.0, 2.0, 3.0, 1.0},
				{1.0, 1.0, 0.0, 3.0, 2.0, 1.0, -1.0, 0.0},
				{4.0, 1.0, 2.0, 3.0, -2.0, 2.0, 2.0, 1.0},
				{-1.0, -3.0, -2.0, 2.0, 0.0, 2.0, 4.0, 1.0}, 
				{3.0, 5.0, -1.0, 1.0, 1.0, 3.0, 0.0, -2.0},
				{1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0}
				};
		
		double[][] partB = {{1.0}, {2.0}, {2.0}, {-2.0}, {1.0}, {7.0}, {14.0}, {6.0}};
		
		Matrix firstM = new Matrix(partA.length, partA[0].length);
		Matrix secondM = new Matrix(partB.length, partB[0].length);
		
		// Assign partA elements to Matrix firstM
		for (int i = 0; i < partA.length; i++) {
			for (int j = 0; j < partA[0].length; j++) {
				firstM.assignElement(partA[i][j], i, j);
			}
		}
		
		// Assign partB elements to Matrix secondM
		for (int i = 0; i < partB.length; i++) {
			for (int j = 0; j < partB[0].length; j++) {
				secondM.assignElement(partB[i][j], i, j);
			}
		}
		
		// Part a: Estimate solution of linear system
		this.gaussJordanC = gaussJordanElimination(firstM, secondM);
		System.out.println("Problem 9" + "\n");
		System.out.println("x = " + gaussJordanC.getElement(0, 8) + "  " + "y = " + gaussJordanC.getElement(1, 8) + "  " + "z = " + gaussJordanC.getElement(2, 8) + "  " +
				"w = " + gaussJordanC.getElement(3, 8) + "  " + "\n" + "a = " + gaussJordanC.getElement(4, 8) + "   " + "b = " + gaussJordanC.getElement(5, 8) + "   " + "c = " + 
				gaussJordanC.getElement(6, 8) + "  " + "d = " + gaussJordanC.getElement(7, 8) + "\n");
		
		// Part b: Compute determinant of coefficient matrix
		this.determinantC = gaussReductionDeterminant(firstM);
		System.out.println("The Determinant of the coefficient Matrix A is: " + determinantC);
		
		// Part c_i: Compute inverse of coefficient matrix
		this.inverseC = matrixInverse(firstM);
		System.out.println("The Inverse of coefficient Matrix A is: " + "\n");
		System.out.println(inverseC);
		
		// Part c_ii: Compute the determinant of inverse matrix 
		this.determinantInverseC = gaussReductionDeterminant(inverseC);
		System.out.println("The Determinant of the Inverse is: " + determinantInverseC);
		
		// Part c_iii: Compute the product of the two determinants
		double determinantProduct = determinantC * determinantInverseC;
		System.out.println("The Product of the Determinants is: " + determinantProduct);
		System.out.println();
		
		// Part d: Validate system solution
		Matrix sysCheck = systemCheck(gaussJordanC);
		System.out.println("System Check");
		System.out.println(sysCheck);
		
		// Problem 10: Condition Number
		conditionNumberC = conditionNumber(firstM, inverseC);
		System.out.println("The Condition Number is: " + conditionNumberC);
		
	}
	
	//------------------------------------------------------CLass Methods----------------------------------------------------------------------
	
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
				System.exit(0);
			}
			
			// Divide row
			augment = rowDivision(i, augment);
			System.out.println(augment);
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
		}
		return augment;
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
	 * The matrixInverse method performs a series of operations on an n-by-n covariance matrix, resulting in the
	 *      inverse of the matrix. 
	 * @param matrixA     an n-by-n covariance matrix
	 * @return inverse     an n-by-n inverse matrix of matrixA
	 */
	private Matrix matrixInverse(Matrix matrixA) {
		
		double error = 1;
		Matrix identity = identityMatrix(matrixA.getRows());
		Matrix augment = augment(matrixA, identity);
		Matrix inverse = new Matrix(identity.getRows(), identity.getColumns());
		
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
				System.exit(0);
			}
			
			// Divide row
			augment = rowDivision(i, augment);
			
			// Elimination step
			for (int j = 0; j < matrixA.getRows(); j++) {
				if (i != j) {
					double value = augment.getElement(j, i);
					for (int k = 0; k < augment.getColumns(); k++) {
						augment.assignElement(augment.getElement(j, k) - (value * augment.getElement(i, k)), j, k); 
					} 
				}
			}		
		}
		
		// Generate inverse matrix
		for (int i = 0; i < identity.getRows(); i++) {
			for (int j = 0; j < identity.getColumns(); j++) {
				inverse.assignElement(augment.getElement(i, identity.getColumns() + j), i, j);
			}
		}
		return inverse;
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
	 * The calculateDiscriminant method calculates the discriminant of a n-by-1 measurement vector, and decides which class
	 *      the point belongs to. 
	 * @param point     an n-by-1 measurement vector
	 * @return vector     1-by-4 vector containing x, y, Class A function value, Class B function value
	 */
	private double[][] calculateDiscriminant(Matrix point) {
		
		// Class 1 discriminant function
		Matrix matrix1A = new Matrix(1, 2); 
		matrix1A.assignElement(point.getElement(0, 0) - meanClass1.getElement(0, 0), 0, 0);
		matrix1A.assignElement(point.getElement(1, 0) - meanClass1.getElement(1, 0), 0, 1);
			
		Matrix first1 = multiply(matrix1A, inverseClass1);
		
		Matrix matrix1B = new Matrix(2, 1);
		matrix1B.assignElement(point.getElement(0, 0) - meanClass1.getElement(0, 0), 0, 0);
		matrix1B.assignElement(point.getElement(1, 0) - meanClass1.getElement(1, 0), 1, 0);
		
		Matrix second1 = multiply(first1, matrix1B);
		double firstThird1 = -0.5 * second1.getElement(0, 0);
			
		double secondThird1 = firstThird1 - (0.5 * Math.log(determinant1));
		double thirdThird1 = secondThird1 + Math.log(pWI/(pWI*2));
		
		// Class 2 discriminant function
		Matrix matrix2A = new Matrix(1, 2); 
		matrix2A.assignElement(point.getElement(0, 0) - meanClass2.getElement(0, 0), 0, 0);
		matrix2A.assignElement(point.getElement(1, 0) - meanClass2.getElement(1, 0), 0, 1);
			
		Matrix first2 = multiply(matrix2A, inverseClass2);
		
		Matrix matrix2B = new Matrix(2, 1);
		matrix2B.assignElement(point.getElement(0, 0) - meanClass2.getElement(0, 0), 0, 0);
		matrix2B.assignElement(point.getElement(1, 0) - meanClass2.getElement(1, 0), 1, 0);
		
		Matrix second2 = multiply(first2, matrix2B);
		double firstThird2 = -0.5 * second2.getElement(0, 0);
			
		double secondThird2 = firstThird2 - (0.5 * Math.log(determinant2));
		double thirdThird2 = secondThird2 + Math.log(pWI/(pWI*2));
		
		double[][] vector = new double[1][4];
		vector[0][0] = point.getElement(0, 0);
		vector[0][1] = point.getElement(1, 0);
		vector[0][2] = thirdThird1;
		vector[0][3] = thirdThird2;
		
		return vector;
	}
	
	/**
	 * The createBoundary method generates a list of (x, y) coordinates whose difference of discriminants are equal to zero. 
	 * @param pointsA	ArrayList of matrices 
	 * @param pointsB	ArrayList of matrices
	 * @return boundaryPoints	ArrayList of matrices	 */
	private ArrayList<Matrix> createBoundary(ArrayList<Matrix> pointsA, ArrayList<Matrix> pointsB) {
		
		
		double maxX = 0; double minX = 0;
		double maxY = 0; double minY = 0;
		
		for (int i = 0; i < pointsA.size(); i++) {
			
			// Find max and minimum x-values
			if (pointsA.get(i).getElement(0, 0) > maxX) {
				maxX = pointsA.get(i).getElement(0, 0);
			}
			else if (pointsA.get(i).getElement(0, 0) < minX) {
				minX = pointsA.get(i).getElement(0, 0);
			}
			if (pointsB.get(i).getElement(0, 0) > maxX) {
				maxX = pointsB.get(i).getElement(0, 0);
			}
			else if (pointsB.get(i).getElement(0, 0) < minX) {
				minX = pointsB.get(i).getElement(0, 0);
			}
			
			// Find max and minimum y-values
			if (pointsA.get(i).getElement(1, 0) > maxY) {
				maxY = pointsA.get(i).getElement(1, 0);
			}
			else if (pointsA.get(i).getElement(0, 0) < minY) {
				minY =pointsA.get(i).getElement(0, 0);
			}
			if (pointsB.get(i).getElement(1, 0) > maxY) {
				maxY = pointsB.get(i).getElement(1, 0);
			}
			else if (pointsB.get(i).getElement(1, 0) < minY) {
				minY = pointsB.get(i).getElement(1, 0);
			}
		}
		
		double lastX = minX; double lastY = minY;
		ArrayList<Matrix> boundaryPoints = new ArrayList<>();
		
		for (double i = minX; i < maxX; i += 0.01) {
			for (double j = minY; j < maxY; j += 0.01) {
				Matrix m = new Matrix(2, 1);
				m.assignElement(i, 0, 0);
				m.assignElement(j, 1, 0);
				double[][] value = calculateDiscriminant(m);
				if (Math.abs(Math.abs(value[0][2]) - Math.abs(value[0][3])) < 0.01 || Math.abs(Math.abs(value[0][3]) - Math.abs(value[0][2])) < 0.01) {
					double x = value[0][0];
					double y = value[0][1];
					if (x != lastX && y != lastY) {
						lastX = x; lastY = y;
						boundaryPoints.add(m);
					}
				}
			}
		}	
		return boundaryPoints;
	}
	
	/**
	 * The conditionNumber method computes the condition number for a coefficient matrix and its inverse  
	 * @param coefficient     an n-by-n coefficient matrix
	 * @param inverse     an n-by-b inverse matrix
	 * @return conditionNumber     the condition number as a double 
	 */
	private double conditionNumber(Matrix coefficient, Matrix inverse) {
		
		double coefficientMax = 0;
		double inverseMax = 0;
		double conditionNumber;
		
		for (int i = 0; i < coefficient.getRows(); i++) {
			double cTemp = 0;
			double iTemp = 0;
			for (int j = 0; j < coefficient.getColumns(); j++) {
				cTemp += Math.abs(coefficient.getElement(i, j));
				iTemp += Math.abs(inverse.getElement(i, j));
			}
			if (cTemp > coefficientMax) {
				coefficientMax = cTemp;
			}
			if (iTemp > inverseMax) {
				inverseMax = iTemp;
			}
		}
		conditionNumber = coefficientMax * inverseMax;
		return conditionNumber;
	}
	
	/**
	 * The systemCheck method verifies the calculated b-values for an n x n linear system, by multiplying the n x n inverse matrix
	 * 		of the linear system by the b-values; producing the original x-values. 
	 * @param coefficient     a solved augmented coefficient n x n matrix
	 * @return xValues    a n x 1 matrix containing x-values
	 */
	private Matrix systemCheck(Matrix coefficient) {
	
		Matrix bValues = new Matrix(coefficient.getRows(), 1);
		
		for (int i = 0; i < coefficient.getRows(); i++) {
			bValues.assignElement(coefficient.getElement(i, (coefficient.getColumns() - 1)), i, 0);
		}
		
		Matrix identityB = gaussJordanElimination(inverseC, bValues);
		System.out.println(identityB);
		
		Matrix xValues = new Matrix(identityB.getRows(), 1);
		
		for (int i = 0; i < identityB.getRows(); i++) {
			xValues.assignElement(identityB.getElement(i, (identityB.getColumns() - 1)), i, 0);
		}
		return xValues;
	}
}



