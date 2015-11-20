package edu.illinois.cs.cogcomp.cureData;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.ObjectOutputStream;
import java.util.Collections;
import java.util.Random;

/**
 * Created by ngupta19 on 11/16/15.
 */
public class Embeddings {

    public static DataEmbeddings de;
    public static double[] unknown_embedding;

    public static void countUnknowns(){
        int unknown = 0, unknown_lower = 0;
        for(String word : de.word_Int_Map.keySet()){
            if(de.word_vectors.isUnknown(word))
                unknown++;

            if(de.word_vectors.isUnknown(word.toLowerCase()))
                unknown_lower++;
        }
        System.out.println("Case Unknown : " + unknown);
        System.out.println("Lower Case Unkown :  " + unknown_lower);

    }

    public static void write_WordInt_Embeddings(String filepath) throws Exception{
        int un = 0;
        BufferedWriter bw = new BufferedWriter(new FileWriter(filepath));
        StringBuffer write_buffer = new StringBuffer();

        for(String word : de.word_Int_Map.keySet()){
            String output = "";
            int int_proxy = de.word_Int_Map.get(word);
            double [] embedding;
            if(!de.word_vectors.isUnknown(word)) {
                embedding = de.word_vectors.lookup(word);
            }
            else {
                //embedding = unknown_embedding;
                embedding = return_unknownEmbed();
                un++;
            }

            output += int_proxy + " ";

            System.out.println(embedding.length);
            for(int i=0; i<de.embedding_size; i++){
                String em = String.valueOf(embedding[i]);
                output += em + " ";
            }
            output = output.substring(0, output.length() - 1);
            output += "\n";
            write_buffer.append(output);
        }
        String output_data = write_buffer.toString();
        output_data = output_data.substring(0, output_data.length() - 1);
        bw.write(output_data);
        bw.close();

        System.out.println("*** \t Embeddings Written. Uknowns : " + un + " \t ***");

    }

    public static void make_unknownEmbed(){
        int size = de.embedding_size;
        unknown_embedding = new double[size];
        Random rand = new Random();
        for(int i=0; i<size; i++){
            unknown_embedding[i] = rand.nextDouble() - 0.5;
        }
    }

    public static double[] return_unknownEmbed(){
        int size = de.embedding_size;
        double [] embedding = new double[size];
        Random rand = new Random();
        for(int i=0; i<size; i++){
            embedding[i] = rand.nextDouble() - 0.5;
        }
        return embedding;
    }


    public static void main(String [] args ) throws Exception{
        de = new DataEmbeddings();
        make_unknownEmbed();

        write_WordInt_Embeddings("data/wordint_embeddings.txt");


    }
}
