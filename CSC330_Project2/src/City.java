
public class City {

	
	// Class Properties:
	String label;
	double xPos; double yPos;
	
	public City(String label, double xPos, double yPos) {
		this.label = label;
		this.xPos = xPos;
		this.yPos = yPos;
	}
	
	public double getX() {
		return xPos;
	}
	
	public double getY() {
		return yPos;
	}
	
	public String getLabel() {
		return label;
	}
	
	public String toString() {
		String cityS = new String();
		cityS = label + " " + xPos +" " + yPos;
		
		return cityS;
	}
	
}
