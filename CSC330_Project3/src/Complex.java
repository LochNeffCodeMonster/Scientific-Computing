import java.util.Objects;

public class Complex {
	
	//Complex properties
	//Real number
    private final double real;  
    //Imaginary number
    private final double imaginary;   

    /**
     * Class constructor that generates new instance of object.
     * @param real	real number
     * @param imag	imaginary number
     */
    public Complex(double real, double imag) {
        this.real = real;
        this.imaginary = imag;
    }

    /**
     * Returns the string representation of Complex object
     */
    public String toString() {
        if (imaginary == 0) return real + " +" + " 0.0i";
        if (real == 0) return imaginary + "i";
        if (imaginary <  0) return real + " - " + (-imaginary) + "i";
        return real + " + " + imaginary + "i";
    }

    /**
     * The magnitude method calculates the magnitude of a Complex object.
     * @return	magnitude
     */
    public double magnitude() {
        return Math.hypot(real, imaginary);
    }

    /**
     * The phase method calculates the phase of a Complex object.
     * @return	phase
     */
    public double phase() {
        return Math.atan2(imaginary, real);
    }

   /**
    * The plus method adds two Complex objects. 
    * @param b	Complex object	
    * @return	new Complex object(this + b)
    */
    public Complex plus(Complex b) {
        Complex a = this;             // invoking object
        double real = a.real + b.real;
        double imag = a.imaginary + b.imaginary;
        return new Complex(real, imag);
    }

   /**
    * The minus method subtracts a Complex object from another.
    * @param b	Complex object
    * @return	new Complex object whos value is (this - b)
    */
    public Complex minus(Complex b) {
        Complex a = this;
        double real = a.real - b.real;
        double imag = a.imaginary - b.imaginary;
        return new Complex(real, imag);
    }

    /**
     * The times method multiplies two Complex objects
     * @param b	Complex object
     * @return	new Complex object whose value is (this * b)
     */
    public Complex times(Complex b) {
        Complex a = this;
        double real = a.real * b.real - a.imaginary * b.imaginary;
        double imag = a.real * b.imaginary + a.imaginary * b.real;
        return new Complex(real, imag);
    }

    /**
     * The scale method multiplies a Complex object by a scalar value
     * @param scale	scalar value
     * @return	new Complex object whose value is (this * scale)
     */
    public Complex scale(double scale) {
        return new Complex(scale * real, scale * imaginary);
    }

    /**
     * The re method returns real number of Complex object
     * @return real	real number
     */
    public double re() { 
    		return real; 
    	}
    
    /**
     * The im methods returns the imaginary number of a Complex object
     * @return imaginary		imaginary number
     */
    public double im() {
    		return imaginary;
    	}

    /**
     * The division method divides a Complex object by a number
     * @param N	integer value	
     * @return	new Complex object whose value is (this / N)
     */
    public Complex division(int N) {
    		Complex a = this;
    		return new Complex(real / N, imaginary / N);
    }

}
