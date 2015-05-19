/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wedt.project;

import cmu.arktweetnlp.POSTagger;
import cmu.arktweetnlp.Token;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.converters.CSVLoader;

/**
 *
 * @author Michał
 */
public class Common {
    
    private ArrayList<String> featureWords;
    private ArrayList<Attribute> attributeList;
    private ArrayList<String> sentimentClassList;
    private POSTagger posTagger;
    
    Common()
    {
        ObjectInputStream ois = null;
        try {
            //reads the feature words list to a hashset
            ois = new ObjectInputStream(new FileInputStream("FeatureWordsList.dat"));
            featureWords = (ArrayList<String>) ois.readObject();
        } catch (Exception ex) {
            System.out.println("Exception in Deserialization");
        } finally {
            try {
                ois.close();
            } catch (IOException ex) {
                System.out.println("Exception while closing file after Deserialization");
            }
        }
        
        //creating an attribute list from the list of feature words
        sentimentClassList = new ArrayList<>();
        sentimentClassList.add("positive");
        sentimentClassList.add("negative");
        for(String featureWord : featureWords)
        {
            attributeList.add(new Attribute(featureWord));
        }
        //the last attribute reprsents ths CLASS (Sentiment) of the tweet
        attributeList.add(new Attribute("Sentiment",sentimentClassList));
    }

    
    public Instances getTrainingDataset(final String INPUT_FILENAME) {
        try{
            //reading the training dataset from CSV file
            CSVLoader trainingLoader = new CSVLoader();
            trainingLoader.setSource(new File(INPUT_FILENAME));
            return trainingLoader.getDataSet();
        }catch(IOException ex)
        {
            System.out.println("Exception in getTrainingDataset Method");
        }
        return null;
    }
    
    
    public Instances createInstances(final String INSTANCES_NAME) {
        
        //create an Instances object with initial capacity as zero 
        Instances instances = new Instances(INSTANCES_NAME,attributeList,0);
        
        //sets the class index as the last attribute (positive or negative)
        instances.setClassIndex(instances.numAttributes()-1);
            
        return instances;
    }
    
    
    public Instance extractFeature(Instance inputInstance) {
        Map<Integer,Double> featureMap = new TreeMap<>();
        List<Token> tokens = posTagger.runPOSTagger(inputInstance.stringValue(0));

        for(Token token : tokens)
        {
            switch(token.getPOS())
            {
                case "A":
                case "V":
                case "R":   
                case "#":   String word = token.getWord().replaceAll("#","");
                            if(featureWords.contains(word))
                            {
                                //adding 1.0 to the featureMap represents that the feature word is present in the input data
                                featureMap.put(featureWords.indexOf(word),1.0);
                            }
            }
        }
        int indices[] = new int[featureMap.size()+1];
        double values[] = new double[featureMap.size()+1];
        int i=0;
        for(Map.Entry<Integer,Double> entry : featureMap.entrySet())
        {
            indices[i] = entry.getKey();
            values[i] = entry.getValue();
            i++;
        }
        indices[i] = featureWords.size();
        values[i] = (double)sentimentClassList.indexOf(inputInstance.stringValue(1));
        return new SparseInstance(1.0,values,indices,featureWords.size());
    }
    
    public Instance extractFeatureFromString(String sentence) {
        Map<Integer,Double> featureMap = new TreeMap<>();
        List<Token> tokens = posTagger.runPOSTagger(sentence);

        for(Token token : tokens)
        {
            switch(token.getPOS())
            {
                case "A":
                case "V":
                case "R":   
                case "#":   String word = token.getWord().replaceAll("#","");
                            if(featureWords.contains(word))
                            {
                                //adding 1.0 to the featureMap represents that the feature word is present in the input data
                                featureMap.put(featureWords.indexOf(word),1.0);
                            }
            }
        }
        int indices[] = new int[featureMap.size()+1];
        double values[] = new double[featureMap.size()+1];
        int i=0;
        for(Map.Entry<Integer,Double> entry : featureMap.entrySet())
        {
            indices[i] = entry.getKey();
            values[i] = entry.getValue();
            i++;
        }
        indices[i] = featureWords.size();
        values[i] = (double)sentimentClassList.indexOf("positive");
        return new SparseInstance(1.0,values,indices,featureWords.size());
    }

}
