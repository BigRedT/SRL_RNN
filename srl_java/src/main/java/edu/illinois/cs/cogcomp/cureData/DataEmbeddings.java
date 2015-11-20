package edu.illinois.cs.cogcomp.cureData;

import edu.illinois.cs.cogcomp.core.io.LineIO;
import edu.illinois.cs.cogcomp.nlp.wordmap.EfficientWordMap;

import java.io.FileNotFoundException;
import java.io.Serializable;
import java.util.HashMap;

/**
 * Created by ngupta19 on 11/16/15.
 */

public class DataEmbeddings {
    public HashMap<Integer, Double[]> wordint_Embedding_Map;
    public HashMap<String, Integer> word_Int_Map;
    public EfficientWordMap word_vectors;
    public int embedding_size = 300;

    public void loadWordVectors(){

        System.out.println("*** \t LOADING WORD VECTORS \t ***");
        try {
            word_vectors = new EfficientWordMap("/shared/shelley/wieting2/illinois-word-embeddings/glove_all_vectors.txt" ,embedding_size, true);
        }
        catch(FileNotFoundException e) {
            System.err.println("NOT FOUND : Glove word vectors file");
        }
        System.out.println("*** \t WORD VECTORS  LOADED \t ***");
    }

    public void loadWordIntMap(String filepath) throws Exception {
        LineIO lineIO = new LineIO();
        String [] data = lineIO.slurp(filepath).split("\n");
        for(int l = 0; l<data.length; l++) {
            String [] word_int = data[l].split(" ");
            String word = word_int[0].trim();
            int in = Integer.parseInt(word_int[1].trim());
            word_Int_Map.put(word, in);
        }
        System.out.println("Words in Vocab : " + word_Int_Map.keySet().size());
    }

    public void setup() throws Exception {
        wordint_Embedding_Map = new HashMap<Integer, Double[]>();
        word_Int_Map = new HashMap<String, Integer>();
        loadWordIntMap("data/word_int_map.txt");
        loadWordVectors();

    }

    public DataEmbeddings() throws Exception{
        setup();
    }

}
