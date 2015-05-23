/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wedt.project;

import java.io.File;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Michał
 */

public class SvmClassifier {
    private Classifier cls;
    
    SvmClassifier() {
        cls = new SMO();
    }
    
    public void train(File file, Common cmn) {
        Instances trainingInstances = cmn.getPrepapredSet(file);
            
        try {
            cls.buildClassifier(trainingInstances);
            weka.core.SerializationHelper.write("SVM.model",cls);
        } catch (Exception ex) {
            System.out.println("Blad uczenia");
            System.out.println(ex.toString());
        }
    }
    
    public String classifySingle(String tweet, Common cmn) {
        Instance instance = cmn.extractFeatureFromString(tweet);
        instance.setDataset(cmn.getEmptyInstances("instances"));
        
        try {
            cls = (Classifier) weka.core.SerializationHelper.read("SVM.model");
            double score = cls.classifyInstance(instance);
            return cmn.sentiment.get((int)score);
        } catch (Exception ex) {
            System.out.println("Blad klasyfikacji Single");
        }
        return null;
    }
    
    public int classifyFromCsv(File file, Common cmn) {
        Instances instances = cmn.getPrepapredSet(file);
            
        try {
            cls = (Classifier) weka.core.SerializationHelper.read("SVM.model");
            int errors = 0;
            
            for(Instance testInstance : instances) {
                double score = cls.classifyInstance(testInstance);
                System.out.println(instances.attribute("Sentiment").value((int)score));
                if (testInstance.value(instances.attribute("Sentiment")) != score)
                    errors++;
            }
            return errors;
        } catch (Exception ex) {
            System.out.println("Blad klasyfikacji CSV");
        }
        return -1;
    }

}
