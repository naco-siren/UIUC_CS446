/**
 * Created by nacos on 2016/9/18.
 */

import java.io.*;

import weka.core.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

/**
 * Created by nacos on 2016/9/18.
 */
public class FeatureExtraction {
    // Info
    private String _fileName;

    // Features configuration
    final private String[] NAME_TYPE = {"firstName", "lastName"};
    final private int POSITION_RANGE = 5;
    final private int INITAL_CAPACITY = 100000;

    // Feature
    private FastVector _attributes;
    private FastVector[][][] _nominalAttrs;
    private FastVector _classAttr;
    private int _attributesCount;

    // Data
    private Instances _instances;


    public FeatureExtraction(String fileName){
        this._fileName = fileName;
    }

    public void extractToFile(String destFileName){
        // Attributes
        _attributes = new FastVector();
        _nominalAttrs = new FastVector[NAME_TYPE.length][POSITION_RANGE][26];
        // Character features
        for(int i = 0; i < NAME_TYPE.length; i++){
            String nameType = NAME_TYPE[i];
            for(int j = 0; j < POSITION_RANGE; j++){
                for(int k = 0; k < 26; k++){
                    // Value enum for this feature
                    _nominalAttrs[i][j][k] = new FastVector();
                    _nominalAttrs[i][j][k].addElement(String.valueOf(1));
                    _nominalAttrs[i][j][k].addElement(String.valueOf(0));

                    // Add this feature to _attributes
                    String attrName =  NAME_TYPE[i] + (j+0) + "=" + (char)('a' + k); //TODO: Check if start position is 1
                    _attributes.addElement(new Attribute(attrName, _nominalAttrs[i][j][k]));
                }
            }
        }
        // Label class
        _classAttr = new FastVector();
        _classAttr.addElement("+");
        _classAttr.addElement("-");
        _attributes.addElement(new Attribute("Class", _classAttr));

        // Instances
        _instances = new Instances("Badges", _attributes, INITAL_CAPACITY);
        _attributesCount = _instances.numAttributes();
        assert(_attributesCount == NAME_TYPE.length * POSITION_RANGE * 26 + 1);
        _instances.setClassIndex(_attributesCount - 1);


        // Read from files
        File file = new File(_fileName);
        BufferedReader bufferedReader;
        try{
            bufferedReader = new BufferedReader(
                    new InputStreamReader(new FileInputStream(file)));

            int instancesCount = 0;
            String line;
            while((line = bufferedReader.readLine()) != null){
                System.out.println(line);

                // Parse instance
                String parts[] = line.split(" ");
                char labelChar = parts[0].charAt(0);
                char[] firstNameChars = parts[1].toCharArray();
                char[] lastNameChars = parts[2].toCharArray();


                // A new instance
                double[] vals = new double[_instances.numAttributes()];

                // First name
                for(int i = 0; i < POSITION_RANGE; i++){
                    // This position's attributes are all default "0"
                    for(int j = 0; j < 26; j++){
                        vals[0*POSITION_RANGE*26 + i*26 + j] = _nominalAttrs[0][i][j].indexOf("0");
                    }

                    // Decide this position's attribute if it has a char
                    if(i < firstNameChars.length){
                        int charIndex = firstNameChars[i] - 'a';
                        vals[0*POSITION_RANGE*26 + i*26 + charIndex] = _nominalAttrs[0][i][charIndex].indexOf("1");
                    }
                }

                // Last name
                for(int i = 0; i < POSITION_RANGE; i++){
                    // This position's attributes are all default "0"
                    for(int j = 0; j < 26; j++){
                        vals[1*POSITION_RANGE*26 + i*26 + j] = _nominalAttrs[1][i][j].indexOf("0");
                    }

                    // Decide this position's attribute if it has a char
                    if(i < lastNameChars.length){
                        int charIndex = lastNameChars[i] - 'a';
                        vals[1*POSITION_RANGE*26 + i*26 + charIndex] = _nominalAttrs[1][i][charIndex].indexOf("1");
                    }
                }

                // Label
                vals[2*POSITION_RANGE*26] = labelChar == '+'? _classAttr.indexOf("+") : _classAttr.indexOf("-");

                _instances.add(new DenseInstance(1.0, vals)); // TODO: What instance type?

                instancesCount++;
            }

            // Save Instances into .ARFF file
            assert(instancesCount == _instances.size());
            ArffSaver arffSaver = new ArffSaver();
            arffSaver.setInstances(_instances);
            arffSaver.setFile(new File("./" + destFileName));
            arffSaver.writeBatch();

            System.out.println("Altogether " + instancesCount + " instances are saved!");
        } catch (Exception e){
            e.printStackTrace();
        }


    }

}
