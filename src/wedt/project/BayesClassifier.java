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
    
    BayesClassifier() {
        cls = new NaiveBayes();
    }
    
    public void train(File file, Common cmn) {
        Instances instances = cmn.getPrepapredSet(file);
                
        try {
            cls.buildClassifier(instances);
            weka.core.SerializationHelper.write("Bayes.model",cls);
        } catch (Exception ex) {
            System.out.println("Blad uczenia");
            System.out.println(ex.toString());
        }
    }
    
    public String classifySingle(String tweet, Common cmn) {
        System.out.println(tweet);
        Instance instance = cmn.extractFeatureFromString(tweet);
        instance.setDataset(cmn.getEmptyInstances("instances"));
        
        try {
            cls = (Classifier) weka.core.SerializationHelper.read("Bayes.model");
            double score = cls.classifyInstance(instance);
            double dist[] = cls.distributionForInstance(instance); // dokladne dane
            for (int i = 0; i < dist.length; i++)
                System.out.println(dist[i] + "");
            return cmn.sentiment.get((int)score);
        } catch (Exception ex) {
            System.out.println("Blas klasyfikacji Single");
        }
        return null;
    }
    
    public int classifyFromCsv(File file, Common cmn) {   
        Instances instances = cmn.getPrepapredSet(file);
            
        try {
            cls = (Classifier) weka.core.SerializationHelper.read("Bayes.model");
            int errors = 0;

            for(Instance testInstance : instances) {
                double score = cls.classifyInstance(testInstance);
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
