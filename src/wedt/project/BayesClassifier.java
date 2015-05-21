/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wedt.project;

import java.io.File;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Michał
 */

public class BayesClassifier {
    private Classifier cls;
    private Common cmn;
    
    BayesClassifier() {
        cls = new NaiveBayes();
    }
    
    public void train(File file) {
        Instances instances = cmn.getPrepapredSet(file);
                
        try {
            cls.buildClassifier(instances);
            weka.core.SerializationHelper.write("Bayes.model",cls);
        } catch (Exception ex) {
            System.out.println("Blad uczenia");
        }
    }
    
    public String classifySingle(String tweet) {
        Instance instance = cmn.extractFeatureFromString(tweet);
        
        try {
            cls = (Classifier) weka.core.SerializationHelper.read("Bayes.model");
            double score = cls.classifyInstance(instance);
            return cmn.sentiment.get((int)score);
        } catch (Exception ex) {
            System.out.println("Blas klasyfikacji Single");
        }
        return null;
    }
    
    public int classifyFromCsv(File file) {   
        Instances instances = cmn.getPrepapredSet(file);
            
        try {
            cls = (Classifier) weka.core.SerializationHelper.read("Bayes.model");
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
