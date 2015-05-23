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
    
    private ArrayList<Attribute> attributes;
    public ArrayList<String> sentiment;
    private ArrayList<String> featureWords;
    private POSTagger tagger;
    
    Common() {
        attributes = new ArrayList<>();
        sentiment = new ArrayList<>();
        sentiment.add("positive");
        sentiment.add("negative");
        tagger = new POSTagger();
        
        ObjectInputStream inputS = null;
        try {
            inputS = new ObjectInputStream(new FileInputStream("featureWords.dat"));
            featureWords = (ArrayList<String>) inputS.readObject();
            inputS.close();
        } catch (Exception ex) {
            System.out.println("Blad wczytywania pliku z feature words");
        }
        
        for(String featureWord : featureWords) {
            attributes.add(new Attribute(featureWord));
        }
        attributes.add(new Attribute("Sentiment",sentiment));
    }
    
    public Instances getPrepapredSet(File file) {
        try {
            CSVLoader csvLoader = new CSVLoader();
            csvLoader.setSource(file);
            Instances loadedInstances = csvLoader.getDataSet();
            Instances instances = getEmptyInstances("instances");
            
            for(Instance currentInstance : loadedInstances) {
                Instance tmpInstance = extractFeature(currentInstance);
                tmpInstance.setDataset(instances);
                instances.add(tmpInstance);
            }
     
            return instances;
        } catch (IOException e) {
            System.out.println("Blad w przygotowywaniu zbioru");
            System.out.println(e.toString());
        }
        
        return null;
    }    
    
    public Instances getEmptyInstances(final String name) {
        
        Instances instances = new Instances(name,attributes,0);
        instances.setClassIndex(instances.numAttributes()-1);
        return instances;
    }
       
    public Instance extractFeature(Instance input) {
        Map<Integer,Double> map = new TreeMap<>();
        List<Token> tokens = tagger.runPOSTagger(input.stringValue(0));

        for(Token token : tokens) {
            switch(token.getPOS()) {
                case "A":
                case "V":
                case "R":   
                case "#":   
                    String word = token.getWord().replaceAll("#","");
                    if(featureWords.contains(word)) {
                        map.put(featureWords.indexOf(word),1.0);
                    }
            }
        }
        int indices[] = new int[map.size()+1];
        double values[] = new double[map.size()+1];
        int i=0;
        for(Map.Entry<Integer,Double> entry : map.entrySet()) {
            indices[i] = entry.getKey();
            values[i] = entry.getValue();
            i++;
        }
        indices[i] = featureWords.size();
        values[i] = (double)sentiment.indexOf(input.stringValue(1));
        return new SparseInstance(1.0,values,indices,featureWords.size());
    }
    
    public Instance extractFeatureFromString(String sentence) {
        Map<Integer,Double> map = new TreeMap<>();
        List<Token> tokens = tagger.runPOSTagger(sentence);

        for(Token token : tokens) {
            switch(token.getPOS()) {
                case "A":
                case "V":
                case "R":   
                case "#":   
                    String word = token.getWord().replaceAll("#","");
                    if(featureWords.contains(word)) {
                        map.put(featureWords.indexOf(word),1.0);
                    }
            }
        }
        int indices[] = new int[map.size()+1];
        double values[] = new double[map.size()+1];
        int i=0;
        for(Map.Entry<Integer,Double> entry : map.entrySet()) {
            indices[i] = entry.getKey();
            values[i] = entry.getValue();
            i++;
        }
        indices[i] = featureWords.size();
        values[i] = (double)sentiment.indexOf("neutral");
        return new SparseInstance(1.0,values,indices,featureWords.size());
    }

}
