
public class Matrix {
	
	/**
	 * Class properties:
	 * matrix	2d array representation of a matrix.
	 * rows	 	number of rows in a matrix.
	 * columns	number of columns in a matrix.
	 */
	double[][] matrix;
	int rows;
	int columns;
	
	/**
	 * Class constructor instantiates a matrix of a specified dimension
	 * @param matrix		2d array of a specified dimension 
	 */
	public Matrix(double[][] matrix) {
		this.matrix = matrix;
		this.columns = matrix[0].length;
		this.rows = matrix.length;
	}
	
	/**
	 * Class constructor instantiates a matrix of a specified dimension
	 * @param rows	integer indicating number of rows
	 * @param columns    integer indicating number of columns
	 */
	public Matrix(int rows, int columns) {
		this.rows = rows;
		this.columns = columns;
		matrix = new double[rows][columns];
		
	}
	
	/**
	 * The getRows method gets the number of rows of a matrix.
	 * @return rows    the number of rows of a matrix.
	 */
	public int getRows() {
		return rows;
	}
	/**
	 * The getColumns method gets the number of columns of a matrix.
	 * @return columns    the number of columns of a matrix.
	 */
	public int getColumns() {
		return columns;
	}
	/**
	 * The getMatrix method gets the 2D array representation of a matrix.
	 * @return matrix    a 2d array representation of a matrix. 
	 */
	public double[][] getMatrix() {
		return matrix;
	}
	/**
	 * The getElement method gets an element at a specified location in a matrix.
	 * @param i    integer corresponding to a row in a matrix.
	 * @param j	   integer corresponding to a column in a matrix.
	 * @return matrix[i][j]    an element at row "i" and column "j" in a matrix.
	 */
	public double getElement(int i, int j) {
		return matrix[i][j];
	}
	/**
	 * The assignElement method assigns an element at a specified location in a matrix. 
	 * @param value	integer to be assigned.
	 * @param i    integer corresponding to a row in a matrix
	 * @param j    integer corresponding to a column in a matrix.
	 */
	public void assignElement(double value, int i, int j) {
		this.matrix[i][j] = value;
	}
	/**
	 * The toString method returns a string representation of a matrix. 
	 */
	public String toString() {
		System.out.println(rows + "x" + columns + " matrix");
		String matrixString = new String();
		
		for (int i = 0; i < rows; i++) {
		    for (int j = 0; j < columns; j++) {
		    		matrixString += Double.toString(matrix[i][j]) + " ";
		    }
		    matrixString += "\n";
		}
		return matrixString;	
	}
	
}
