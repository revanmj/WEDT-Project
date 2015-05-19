/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wedt.project;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;

import weka.core.Instance;
import weka.core.Instances;


/**
 *
 * @author Michał
 */
public class SvmClassifier {
    private Instances inputDataset;
    private Classifier classifier;
    private Common cmn;
    
    SvmClassifier(Instances dataset) {
        inputDataset = dataset;
        classifier = new SMO();
    }
    
    public void trainClassifier(final String INPUT_FILENAME, Instances inputDataset)
    {
            cmn = new Common();
            cmn.getTrainingDataset(INPUT_FILENAME);
            
            //trainingInstances consists of feature vector of every input
            Instances trainingInstances = cmn.createInstances("TRAINING_INSTANCES");
            
            for(Instance currentInstance : inputDataset)
            {
                //extractFeature method returns the feature vector for the current input
                Instance currentFeatureVector = cmn.extractFeature(currentInstance);
                
                //Make the currentFeatureVector to be added to the trainingInstances
                currentFeatureVector.setDataset(trainingInstances);
                trainingInstances.add(currentFeatureVector);
            }
            
        try {
            //classifier training code
            classifier.buildClassifier(trainingInstances);
            
            //storing the trained classifier to a file for future use
            weka.core.SerializationHelper.write("SVM.model",classifier);
        } catch (Exception ex) {
            System.out.println("Exception in training the classifier.");
        }
    }
    
    public void testClassifier(final String INPUT_FILENAME)
    {
        cmn.getTrainingDataset(INPUT_FILENAME);
            
        //trainingInstances consists of feature vector of every input
        Instances testingInstances = cmn.createInstances("TESTING_INSTANCES");

        for(Instance currentInstance : inputDataset)
        {
            //extractFeature method returns the feature vector for the current input
            Instance currentFeatureVector = cmn.extractFeature(currentInstance);

            //Make the currentFeatureVector to be added to the trainingInstances
            currentFeatureVector.setDataset(testingInstances);
            testingInstances.add(currentFeatureVector);
        }
            
            
        try {
            //Classifier deserialization
            classifier = (Classifier) weka.core.SerializationHelper.read("SVM.model");
            
            //classifier testing code
            for(Instance testInstance : testingInstances)
            {
                double score = classifier.classifyInstance(testInstance);
                System.out.println(testingInstances.attribute("Sentiment").value((int)score));
            }
        } catch (Exception ex) {
            System.out.println("Exception in testing the classifier.");
        }
    }
    
    public double classifySingle(String tweet) {
        Instance instance = cmn.extractFeatureFromString(tweet);
        
        try {
            classifier = (Classifier) weka.core.SerializationHelper.read("NaiveBayes.model");
            double score = classifier.classifyInstance(instance);
            return score;
        } catch (Exception ex) {
            System.out.println("Exception in testing the classifier.");
        }
        return -1;
    }
}
