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
    
    public int setParameters(String params) {
        try {
            ((NaiveBayes)cls).setOptions(weka.core.Utils.splitOptions(params));
            return 0;
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return -1;
        }
    }
    
    public void train(File file, Common cmn) {
        Instances instances = cmn.getPrepapredSet(file, 0);
                
        try {
            cls.buildClassifier(instances);
            weka.core.SerializationHelper.write("Bayes.model",cls);
        } catch (Exception e) {
            System.out.println("Blad uczenia Bayes");
        }
    }
    
    public String classifySingle(String tweet, Common cmn) {
        System.out.println(tweet);
        Instance instance = cmn.extractFeatureFromString(tweet, 0);
        instance.setDataset(cmn.getEmptyInstances("instances"));
        
        try {
            cls = (Classifier) weka.core.SerializationHelper.read("Bayes.model");
            double score = cls.classifyInstance(instance);
            double dist[] = cls.distributionForInstance(instance); // dokladne dane
            for (int i = 0; i < dist.length; i++)
                System.out.println(dist[i] + "");
            return cmn.sentiment.get((int)score);
        } catch (Exception e) {
            System.out.println("Blas klasyfikacji single Bayes");
        }
        return null;
    }
    
    public int classifyFromCsv(File file, Common cmn) {   
        Instances instances = cmn.getPrepapredSet(file, 0);
        System.out.println("==== Bayes ====");
            
        try {
            cls = (Classifier) weka.core.SerializationHelper.read("Bayes.model");
            int errors = 0;

            for(Instance instance : instances) {
                double score = cls.classifyInstance(instance);
                if (instance.value(instances.attribute("Sentiment")) != score)
                    errors++;
                double dist[] = cls.distributionForInstance(instance);
                cmn.printDetailedResults(instance.value(instances.attribute("Sentiment")), dist, score);
                System.out.println();
            }
            return errors;
        } catch (Exception e) {
            System.out.println("Blad klasyfikacji CSV Bayes");
        }
        return -1;
    }

}
