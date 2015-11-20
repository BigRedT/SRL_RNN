package edu.illinois.cs.cogcomp.cureData;

import edu.illinois.cs.cogcomp.core.io.LineIO;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.HashMap;

/**
 * Created by ngupta19 on 11/16/15.
 *
 * Reads train dev and test data from data/xx.txt and
 *  Creates and writes a word -> int map
 *  Creates and writes a label -> int map
 *
 */

public class Main {
    public static String trainpath = "data/train.txt";
    public static String testpath = "data/test.txt";
    public static String devpath = "data/dev.txt";

    public static HashMap<String, Integer> word_Int_Map;
    public static HashMap<String, Integer> label_Int_Map;


    public static int word_count  = -1;
    public static int label_count  = -1;


    public static void populate_Int_Maps(String filepath) throws Exception {
        LineIO lineIO = new LineIO();
        String data = lineIO.slurp(filepath);
        String [] data_tokens = data.split("\\s+");

        for(int t = 0; t < data_tokens.length; t++){
            String [] word_label = data_tokens[t].split("_");
            String word = word_label[0].trim();
            String label = word_label[1].trim();

            if(!word_Int_Map.containsKey(word)){
                word_Int_Map.put(word, ++word_count);
            }

            if(!label_Int_Map.containsKey(label)){
                label_Int_Map.put(label, ++label_count);
            }
        }
    }

    public static void write_Label_Int_Map(String filepath) throws Exception {
        BufferedWriter bw = new BufferedWriter(new FileWriter(filepath));
        StringBuffer output = new StringBuffer();

        for (String label : label_Int_Map.keySet()) {
            int l_proxy = label_Int_Map.get(label);
            String out = label + " " + l_proxy + "\n";
            output.append(out);
        }
        String write_data = output.toString();
        write_data = write_data.substring(0, write_data.length()-1); // To remove the last stranded "\n"
        bw.write(write_data);
        bw.close();
    }

    public static void write_Word_Int_Map(String filepath) throws Exception {
        BufferedWriter bw = new BufferedWriter(new FileWriter(filepath));
        StringBuffer output = new StringBuffer();

        for (String word : word_Int_Map.keySet()) {
            int w_proxy = word_Int_Map.get(word);
            String out = word + " " + w_proxy + "\n";
            output.append(out);
        }
        String write_data = output.toString();
        write_data = write_data.substring(0, write_data.length()-1); // To remove the last stranded "\n"
        bw.write(write_data);
        bw.close();
    }



    public static void main(String [] args) throws Exception{

        word_Int_Map = new HashMap<String, Integer>();
        label_Int_Map = new HashMap<String, Integer>();

        populate_Int_Maps(trainpath);
        populate_Int_Maps(devpath);
        populate_Int_Maps(testpath);

        write_Label_Int_Map("data/label_int_map.txt");
        write_Word_Int_Map("data/word_int_map.txt");

        System.out.println("Word Count : " + word_count);
        System.out.println("Label Count : " + label_count);

    }








}
