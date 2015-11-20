package edu.illinois.cs.cogcomp.cureData;

import edu.illinois.cs.cogcomp.core.io.LineIO;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.HashMap;

/**
 * Created by ngupta19 on 11/16/15.
 */
public class IntData {
    public static String trainpath = "data/train.txt";
    public static String testpath = "data/test.txt";
    public static String devpath = "data/dev.txt";

    public static String int_trainpath = "data/train_int.txt";
    public static String int_testpath = "data/test_int.txt";
    public static String int_devpath = "data/dev_int.txt";


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

    public static void write_IntData(String in_filepath, String out_filepath) throws Exception {
        BufferedWriter bw = new BufferedWriter(new FileWriter(out_filepath));
        StringBuffer output = new StringBuffer();


        LineIO lineIO = new LineIO();
        String data = lineIO.slurp(in_filepath);
        String [] sens = data.split("\n");
        //String [] data_tokens = data.split("\\s+");
        for(int s = 0; s < sens.length; s++){
            output.append("{" + "\n");
            String [] data_tokens = sens[s].trim().split(" ");

            for(int t = 0; t < data_tokens.length; t++) {
                String[] word_label = data_tokens[t].split("_");
                String word = word_label[0].trim();
                String label = word_label[1].trim();
                int w_proxy = word_Int_Map.get(word);
                int l_proxy = label_Int_Map.get(label);
                output.append(w_proxy + " " + l_proxy + "\n");
            }
            output.append("}" + "\n");
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

        write_IntData(trainpath, int_trainpath);
        write_IntData(testpath, int_testpath);
        write_IntData(devpath, int_devpath);

    }

}
