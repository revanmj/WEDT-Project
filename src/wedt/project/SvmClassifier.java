/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wedt.project;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
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
    
    public List<Integer> classifyFromCsv(File file, Common cmn) {
        Instances instances = cmn.getPrepapredSet(file);
            
        try {
            cls = (Classifier) weka.core.SerializationHelper.read("SVM.model");
            int errAll = 0, errPosNeu = 0, errPosNeg = 0, errNegPos = 0, errNegNeu = 0, errNeuPos = 0, errNeuNeg = 0, i = 0;
            List<Integer> errors = new ArrayList<Integer>();
            
            for(Instance instance : instances) {
                i++;
                double score = cls.classifyInstance(instance);
                double shouldBe = instance.value(instances.attribute("Sentiment"));
                if (shouldBe != score) {
                    errAll++;
                    if (shouldBe == 0.0 && score == 2.0)
                        errPosNeu++;
                    else if (shouldBe == 0.0 && score == 1.0)
                        errPosNeg++;
                    else if (shouldBe == 1.0 && score == 0.0)
                        errNegPos++;
                    else if (shouldBe == 1.0 && score == 2.0)
                        errNegNeu++;
                    else if (shouldBe == 2.0 && score == 0.0)
                        errNeuPos++;
                    else if (shouldBe == 2.0 && score == 1.0)
                        errNeuNeg++;
                }
                //System.out.println("==== SVM ====");
                //double dist[] = cls.distributionForInstance(instance);
                //System.out.print(i + ": ");
                //cmn.printDetailedResults(instance.value(instances.attribute("Sentiment")), dist, score);
                //System.out.println();
            }
            errors.add(errAll); errors.add(errPosNeu); errors.add(errPosNeg); errors.add(errNegPos); errors.add(errNegNeu); errors.add(errNeuPos); errors.add(errNeuNeg);
            return errors;
        } catch (Exception e) {
            System.out.println("Blad klasyfikacji CSV SVM");
        }
        return null;
    }

}
