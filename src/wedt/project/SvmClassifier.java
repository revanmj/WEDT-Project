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
        Instances instances = cmn.getPrepapredSet(file);
            
        try {
            cls.buildClassifier(instances);
            weka.core.SerializationHelper.write("SVM.model",cls);
        } catch (Exception e) {
            System.out.println("Blad uczenia SVM");
            e.printStackTrace();
        }
    }
    
    public String classifySingle(String tweet, Common cmn) {
        try {
            System.out.println("==== SVM ====");
            cls = (Classifier) weka.core.SerializationHelper.read("SVM.model");
            Instances instances = cmn.prepareSingle(tweet);
            double score = cls.classifyInstance(instances.firstInstance());
            double dist[] = cls.distributionForInstance(instances.firstInstance()); // dokladne dane
            System.out.println("dist: " + dist[0] + " " + dist[1] + " " + dist[2]);
            return cmn.sentiment.get((int)score);
        } catch (Exception e) {
            System.out.println("Blad klasyfikacji single SVM");
            e.printStackTrace();
        }
        return null;
    }
    
    public int classifyFromCsv(File file, Common cmn) {
        Instances instances = cmn.getPrepapredSet(file);
        System.out.println("==== SVM ====");
            
        try {
            cls = (Classifier) weka.core.SerializationHelper.read("SVM.model");
            int errors = 0, i = 0;
            
            for(Instance instance : instances) {
                i++;
                double score = cls.classifyInstance(instance);
                if (instance.value(instances.attribute("Sentiment")) != score)
                    errors++;
                double dist[] = cls.distributionForInstance(instance);
                System.out.print(i + ": ");
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
