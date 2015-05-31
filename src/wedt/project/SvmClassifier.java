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
        try {
            ((SMO)cls).setOptions(weka.core.Utils.splitOptions("-M"));
            ((SMO)cls).setBuildLogisticModels(true);
        } catch (Exception e) {}
    }
    
    public int setParameters(String params) {
        try {
            ((SMO)cls).setOptions(weka.core.Utils.splitOptions(params));
            return 0;
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return -1;
        }
    }
    
    public void train(File file, Common cmn) {
        Instances instances = cmn.getPrepapredSet(file, 1);
            
        try {
            cls.buildClassifier(instances);
            weka.core.SerializationHelper.write("SVM.model",cls);
        } catch (Exception e) {
            System.out.println("Blad uczenia SVM");
            e.printStackTrace();
        }
    }
    
    public String classifySingle(String tweet, Common cmn) {
        System.out.println(tweet);
        Instance instance = cmn.extractFeatureFromString(tweet, 1);
        instance.setDataset(cmn.getEmptyInstances("instances"));
        
        try {
            cls = (Classifier) weka.core.SerializationHelper.read("SVM.model");
            double score = cls.classifyInstance(instance);
            double dist[] = cls.distributionForInstance(instance); // dokladne dane
            for (int i = 0; i < dist.length; i++)
                System.out.println(dist[i] + "");
            return cmn.sentiment.get((int)score);
        } catch (Exception e) {
            System.out.println("Blad klasyfikacji single SVM");
            e.printStackTrace();
        }
        return null;
    }
    
    public int classifyFromCsv(File file, Common cmn) {
        Instances instances = cmn.getPrepapredSet(file, 1);
        System.out.println("==== SVM ====");
            
        try {
            cls = (Classifier) weka.core.SerializationHelper.read("SVM.model");
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
            System.out.println("Blad klasyfikacji CSV SVM");
        }
        return -1;
    }

}
