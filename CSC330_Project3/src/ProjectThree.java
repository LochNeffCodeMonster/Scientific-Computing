
import java.util.ArrayList;

public class ProjectThree {
	
	public static void main(String[] args) {
		ProjectThree p3 = new ProjectThree();
	}
	
	public ProjectThree() {
		/*
		//Problem 1i
		double[][] f5Bin = functionF(5, 512);
		for (int i = 0; i < f5Bin.length; i++) {
			System.out.println(f5Bin[i][0]);
		}
		Complex[] f5 = new Complex[f5Bin.length];
		for (int i = 0; i < f5Bin.length; i++) {
			Complex c = new Complex(f5Bin[i][0], 0);
			f5[i] = c;
		}
		Complex[] f5Data = computeFFT(f5, 1);
		double[] f5PSD = computePSD(f5Data);
		for (int j = 0; j < f5Data.length / 2; j++) {
			System.out.println(f5PSD[j]);
		}
		
		
		double[][] f10Bin = functionF(10, 512);
		for (int i = 0; i < f10Bin.length; i++) {
			System.out.println(f10Bin[i][0]);
		}
		Complex[] f10 = new Complex[f10Bin.length];
		for (int i = 0; i < f10Bin.length; i++) {
			Complex c = new Complex(f10Bin[i][0], 0);
			f10[i] = c;
		}
		Complex[] f10Data = computeFFT(f10, 1);
		double[] f10PSD = computePSD(f10Data);
		for (int j = 0; j < f10Data.length / 2; j++) {
			System.out.println(f10PSD[j]);
		}
		
		
		double[][] f100Bin = functionF(100, 512);
		for (int i = 0; i < f100Bin.length; i++) {
			System.out.println(f100Bin[i][0]);
		}
		Complex[] f100 = new Complex[f100Bin.length];
		for (int i = 0; i < f100Bin.length; i++) {
			Complex c = new Complex(f100Bin[i][0], 0);
			f100[i] = c;
		}
		Complex[] f100Data = computeFFT(f100, 1);
		double[] f100PSD = computePSD(f100Data);
		for (int j = 0; j < f100Data.length / 2; j++) {
			System.out.println(f100PSD[j]);
		}
		
		
		//Problem1ii
		double[][] g5Bin = functionG(5,512);
		for (int i = 0; i < g5Bin.length; i++) {
			System.out.println(g5Bin[i][0]);
		}
		Complex[] g5 = new Complex[g5Bin.length];
		for (int i = 0; i < g5Bin.length; i++) {
			Complex c = new Complex(g5Bin[i][0], 0);
			g5[i] = c;
		}
		Complex[] g5Data = computeFFT(g5, 1);
		double[] g5PSD = computePSD(g5Data);
		for (int j = 0; j < g5Data.length / 2; j++) {
			System.out.println(g5PSD[j]);
		}
		
		
		double[][] g10Bin = functionG(10,512);
		for (int i = 0; i < g10Bin.length; i++) {
			System.out.println(g10Bin[i][0]);
		}
		Complex[] g10 = new Complex[g10Bin.length];
		for (int i = 0; i < g10Bin.length; i++) {
			Complex c = new Complex(g10Bin[i][0], 0);
			g10[i] = c;
		}
		System.out.println("");
		Complex[] g10Data = computeFFT(g10, 1);
		double[] g10PSD = computePSD(g10Data);
		for (int j = 0; j < g10Data.length / 2; j++) {
			System.out.println(g10PSD[j]);
		}
		
		
		double[][] g100Bin = functionG(100, 512);
		for (int i = 0; i < g100Bin.length; i++) {
			System.out.println(g100Bin[i][0]);
		}
		Complex[] g100 = new Complex[g100Bin.length];
		for (int i = 0; i < g100Bin.length; i++) {
			Complex c = new Complex(g100Bin[i][0], 0);
			g100[i] = c;
		}
		Complex[] g100Data = computeFFT(g100, 1);
		double[] g100PSD = computePSD(g100Data);
		for (int j = 0; j < g100Data.length / 2; j++) {
			System.out.println(g100PSD[j]);
		}

		//Problem 1c
		//f1000
		double[][] f1000Bin = functionF(1000, 512);
		for (int i = 0; i < f1000Bin.length; i++) {
			System.out.println(f1000Bin[i][0]);
		}
		Complex[] f1000 = new Complex[f1000Bin.length];
		for (int i = 0; i < f1000Bin.length; i++) {
			Complex c = new Complex(f1000Bin[i][0], 0);
			f1000[i] = c;
		}
		Complex[] f1000Data = computeFFT(f1000, 1);
		double[] f1000PSD = computePSD(f1000Data);
		for (int j = 0; j < f1000Data.length / 2; j++) {
			//System.out.println(f1000PSD[j]);
		}
		
		//f10000
		double[][] f10000Bin = functionF(10000, 512);
		for (int i = 0; i < f10000Bin.length; i++) {
			System.out.println(f10000Bin[i][0]);
		}
		Complex[] f10000 = new Complex[f10000Bin.length];
		for (int i = 0; i < f10000Bin.length; i++) {
			Complex c = new Complex(f10000Bin[i][0], 0);
			f10000[i] = c;
		}
		Complex[] f10000Data = computeFFT(f10000, 1);
		double[] f10000PSD = computePSD(f10000Data);
		for (int j = 0; j < f10000Data.length / 2; j++) {
			System.out.println(f10000PSD[j]);
		}
	
		//g1000
		double[][] g1000Bin = functionG(1000, 512);
		for (int i = 0; i < g1000Bin.length; i++) {
			System.out.println(g1000Bin[i][0]);
		}
		Complex[] g1000 = new Complex[g1000Bin.length];
		for (int i = 0; i < g1000Bin.length; i++) {
			Complex c = new Complex(g1000Bin[i][0], 0);
			g1000[i] = c;
		}
		Complex[] g1000Data = computeFFT(g1000, 1);
		double[] g1000PSD = computePSD(g1000Data);
		for (int j = 0; j < g1000Data.length / 2; j++) {
			//System.out.println(g1000PSD[j]);
		}
		
		//g10000
		double[][] g10000Bin = functionG(10000, 512);
		for (int i = 0; i < g10000Bin.length; i++) {
			System.out.println(g10000Bin[i][0]);
		}
		Complex[] g10000 = new Complex[g10000Bin.length];
		for (int i = 0; i < g10000Bin.length; i++) {
			Complex c = new Complex(g10000Bin[i][0], 0);
			g10000[i] = c;
		}
		Complex[] g10000Data = computeFFT(g10000, 1);
		double[] g10000PSD = computePSD(g10000Data);
		for (int j = 0; j < g10000Data.length / 2; j++) {
			System.out.println(g10000PSD[j]);
		}
		
		//Problem2
		//v1
		double[][] v1Bin = functionV(1, 512, 1, 0, 11);
		for (int i = 0; i < v1Bin.length; i++) {
			//System.out.println(v1Bin[i][0]);
		}
		
		Complex[] v1 = new Complex[v1Bin.length];
		for (int i = 0; i < v1Bin.length; i++) {
			Complex c = new Complex(v1Bin[i][0], 0);
			v1[i] = c;
		}
		Complex[] v1Data = computeFFT(v1, 1);
		double[] v1PSD = computePSD(v1Data);
		for (int j = 0; j < v1Data.length / 2; j++) {
			System.out.println(v1PSD[j]);
		}
		
		//v2
		double[][] v2Bin = functionV(1, 512, 1, 0, 27);
		for (int i = 0; i < v2Bin.length; i++) {
			//System.out.println(v2Bin[i][0]);
		}
		
		Complex[] v2 = new Complex[v2Bin.length];
		for (int i = 0; i < v2Bin.length; i++) {
			Complex c = new Complex(v2Bin[i][0], 0);
			v2[i] = c;
		}
		Complex[] v2Data = computeFFT(v2, 1);
		double[] v2PSD = computePSD(v2Data);
		for (int j = 0; j < v2Data.length / 2; j++) {
			System.out.println(v2PSD[j]);
		}
		
		double[][] signalX = signalX(v1Bin, v2Bin);
		Complex[] x = new Complex[signalX.length];
		for (int i = 0; i < signalX.length; i++) {
			Complex c = new Complex(signalX[i][0], 0);
			x[i] = c;
		}
		Complex[] xData = computeFFT(x, 1);
		double[] xPSD = computePSD(xData);
		for (int j = 0; j < xData.length / 2; j++) {
			System.out.println(xPSD[j]);
		}
		
		double[][] signalY = signalY(v1Bin, v2Bin);
		Complex[] y = new Complex[signalY.length];
		for (int i = 0; i < signalY.length; i++) {
			Complex c = new Complex(signalY[i][0], 0);
			y[i] = c;
		}
		Complex[] yData = computeFFT(y, 1);
		double[] yPSD = computePSD(yData);
		for (int j = 0; j < yData.length / 2; j++) {
			System.out.println(yPSD[j]);
		}
		
		
		//Problem3
		double[][] f10Bin = functionF(10, 512);
		for (int i = 0; i < f10Bin.length; i++) {
		}
		
		double[][] pulse = new double[512][1];
		int pulse1 = 128;
		for (int i = 0; i < f10Bin.length; i++) {
			if (i == pulse1) {
				pulse[i][0] = 1;
			}
			else {
				pulse[i][0] = f10Bin[i][0];
			}
		}
		Complex[] p = new Complex[pulse.length];
		for (int i = 0; i < pulse.length; i++) {
			Complex c = new Complex(pulse[i][0], 0);
			p[i] = c;
		}
				
		Complex[] pData = computeFFT(p, 1);
		double[] pPSD = computePSD(pData);
		for (int j = 0; j < pData.length / 2; j++) {
			System.out.println(pPSD[j]);
		}
		
		
		
		
		//3b
		//c = 0.25
		double[][] h1Bin = functionH(1, 512, 0.25);
		for (int i = 0; i < h1Bin.length; i++) {
			//System.out.println(h1Bin[i][0]);
		}
		Complex[] h1 = new Complex[h1Bin.length];
		for (int i = 0; i < h1Bin.length; i++) {
			Complex c = new Complex(h1Bin[i][0], 0);
			h1[i] = c;
		}
		Complex[] h1Data = computeFFT(h1, 1);
		double[] h1PSD = computePSD(h1Data);
		for (int j = 0; j < h1Data.length / 2; j++) {
			System.out.println(h1PSD[j]);
		}
		
		*/
		//c = 0.5
		double[][] h2Bin = functionH(1, 512, 0.75);
		for (int i = 0; i < h2Bin.length; i++) {
			//System.out.println(h2Bin[i][0]);
		}
		Complex[] h2 = new Complex[h2Bin.length];
		for (int i = 0; i < h2Bin.length; i++) {
			Complex c = new Complex(h2Bin[i][0], 0);
			h2[i] = c;
		}
		Complex[] h2Data = computeFFT(h2, 1);
		//double[] h2PSD = computePSD(h2Data);
		for (int j = 0; j < h2Data.length / 2; j++) {
			System.out.println(h2Data[j]);
		}
		
		/*
		//c = 0.75
		double[][] h3Bin = functionH(1, 512, 0.75);
		for (int i = 0; i < h3Bin.length; i++) {
			//System.out.println(h3Bin[i][0]);
		}
		Complex[] h3 = new Complex[h3Bin.length];
		for (int i = 0; i < h3Bin.length; i++) {
			Complex c = new Complex(h3Bin[i][0], 0);
			h3[i] = c;
		}
		Complex[] h3Data = computeFFT(h3, 1);
		double[] h3PSD = computePSD(h3Data);
		for (int j = 0; j < h3Data.length / 2; j++) {
			System.out.println(h3PSD[j]);
		}
		
		*/
		
		
		/*
		double[] data = {
		26160.0, 19011.0, 18757.0, 18405.0, 17888.0, 14720.0, 14285.0, 17018.0, 
		18014.0, 17119.0, 16400.0, 17497.0, 17846.0, 15700.0, 17636.0, 17181.0};
		
		Complex[] z = new Complex[data.length];
		
		for (int i = 0; i < data.length; i++) {
			Complex c = new Complex(data[i], 0);
			z[i] = c;
		}
		
		Complex[] x = computeFFT(z, 1);
		
		for (int j = 0; j < x.length; j++) {
			System.out.println(x[j]);
		}
		*/
		
	}
	
	/**
	 * 
	 * @param s
	 * @param q
	 * @return
	 */
    public double[][] functionF(int s, double q) {
    	
    		int counter = 0;
    		double[][] bin = new double[512][1];
    		
    		for (double t = 0.0; t < 1.0; t += 1.0/q) {
    			double sum = 0;
    			for (int k = 1; k <= s; k++) {
    				double inner = ((2 * k) - 1);
    				double outer = (((2 * Math.PI) * (inner)) * t); 
    				double numerator = Math.sin(outer);
    				sum += (numerator / inner);
    			}
    			bin[counter][0] = sum;
    			counter += 1;
    		}
    		/*
    		for(int j = 0; j < bin.length; j++) {
    			System.out.println(bin[j][0]);
    		}
    		*/
    		return bin;
    }
    
    /**
     * 
     * @param s
     * @param q
     */
    public double[][] functionG(int s, double q) {
    	
    		int counter = 0;
    		double[][] bin = new double[512][1];
		
    		for (double t = 0.0; t < 1.0; t += 1.0/q) {
    			double sum = 0;
    			for (int k = 1; k <= s; k++) {
    				
    				double inner = (2 * k);
    				double outer = (((2 * Math.PI) * inner) * t);
    				
				double numerator = Math.sin(outer);
				sum += (numerator / inner);
			}
    			bin[counter][0] = sum;
    			counter += 1;
		}
    		for(int j = 0; j < bin.length; j++) {
    			System.out.println(bin[j][0]);
    		}
    		return bin;
    }
    
    /**
     * 
     * @param s
     * @param q
     * @param a
     * @param c
     * @param f
     * @return
     */
    public double[][] functionV(int s, double q, int a, int c, int f) {
    	
    		int counter = 0;
		double[][] bin = new double[512][1];
		
		for (double t = 0.0; t < 1.0; t += 1.0/q) {
			double sum = 0;
			for (int k = 1; k <= s; k++) {
				
				double inner = t - c;
				double outer = (2 * Math.PI * f) * inner;
				double answer = a * Math.sin(outer);
				sum += answer;
			}
			bin[counter][0] = sum;
			counter += 1;		
		}
		for(int j = 0; j < bin.length; j++) {
		//	System.out.println(bin[j][0]);
		}
		return bin;
    }
    
    /**
     * 
     * @param v1
     * @param v2
     * @return
     */
    public double[][] signalX(double[][] v1, double[][] v2) {
    	
		double[][] bin = new double[512][1];
		
		for (int i = 0; i < v1.length; i++) {
			bin[i][0] = v1[i][0] + v2[i][0];
		}
		return bin;
    }
    
    /**
     * 
     * @param v1
     * @param v2
     * @return
     */
    public double[][] signalY(double[][] v1, double[][] v2) {
    	
    		double[][] bin = new double[512][1];
    		for (int i = 0; i < v1.length; i++) {
    			bin[i][0] = v1[i][0] * v2[i][0];
    		}
    		return bin;
    }
    
    /**
     * 
     * @param s
     * @param q
     * @param c
     * @return
     */
    public double[][] functionH(double s, double q, double c) {
    	
    		int counter = 0;
		double[][] bin = new double[512][1];
		
		for (double t = 0.0; t < 1.0; t += 1.0/q) {
			double sum = 0;
			for (int k = 1; k <= s; k++) {
				
				double inner = t - c;
				double outer = (14 * Math.PI) * inner;
				double answer = Math.sin(outer);
				sum += answer;
				
			}
			bin[counter][0] = sum;
			counter += 1;		
		}
		for(int j = 0; j < bin.length; j++) {
			//System.out.println(bin[j][0]);
		}
		return bin;
    }    
    
    
    /**
     * The compute FFT method is a one-to-one mapping of real-valued functions of time onto
     * complex-valued functions of frequency. 
     * @param z 		vector z contains the samples of the complex digital signal to be transformed,
     * 				and on exit, it contains the transformed signal.
     * @param d		direction code; if d = 1, it computes the forward FFT of z, and when d = -1, 
     * 				it computes the inverse FFT of z. 
     */
    public Complex[] computeFFT(Complex[] z, double d) {
    	
    		Complex t;
    	
    		//Verify the FFT is a power of 2
    		int N = z.length;
    		if (N % 2 != 0) {
                throw new IllegalArgumentException("n is not a power of 2");
         }
    	
    		//(1)
    		double theta = (-2 * Math.PI * d) / N;
    		int r = N / 2;
    		//(2) - FFT calculations
    		for (int i = 1; i < N - 1; i = 2 * i) {
    			//(a)
    			Complex w = new Complex(Math.cos(i*theta), Math.sin(i*theta));
    			//(b)
    			for (int k = 0; k < N - 1; k++) {
    				//(1)
    				Complex u = new Complex(1.0, 0);
    				//(2)
    				for (int m = 0; m < r; m++) {
    					//System.out.println("i=" + i + "; k=" + k + "; m=" + m + "; r=" + r);
    					           t = z[k + m].minus(z[k + m + r]);
    					    z[k + m] = z[k + m].plus(z[k + m + r]);
    					z[k + m + r] = t.times(u);
    					           u = w.times(u);
    				}
    				//(3)
    				k = k - 1 + (2 * r);
    			}
    			//(c)
    			r = r / 2;
    		}
    		/*
    		for (int p = 0; p < z.length; p++) {
    			System.out.println(z[p]);
    		}
    		*/
    		
    		//(3) - Sorting of results using bit reversal
    		for (int i = 0; i < N; i++) {
    			//(a)
    			r = i;
    			int k = 0;
    			//(b)
    			for (int m = 1; m < N - 1; m = 2*m) {
    				//System.out.println("i=" + i + "; k=" + k + "; m=" + m + "; r=" + r);
    				k = (2 * k) + (r % 2);
    				r = r / 2;
    			}
    			//(c)
    			if (k > i) {
    				t = z[i];
    				z[i] = z[k];
    				z[k] = t;
    			}
    			
    		}
    		//(4) - Normalization of the inverse FFT 
    		if (d < 0) {
    			for (int i = 0; i < N - 1; i++) {
    				z[i] = z[i].division(N);
    			}
    		}
    		return z;
    }
    
    /**
     * 
     * @param z
     * @return
     */
    public double[] computePSD(Complex[] z) {
    	
    		double[] psd  = new double[z.length];
    		
    		for (int i = 0; i < z.length; i++) {
    			psd[i] = (z[i].re() * z[i].re()) + (z[i].im() * z[i].im());
    		}
    		return psd;
    }	

}
