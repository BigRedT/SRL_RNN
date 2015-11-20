package edu.illinois.cs.cogcomp;


import edu.illinois.cs.cogcomp.cachingcurator.CachingAnnotator;
import edu.illinois.cs.cogcomp.core.datastructures.ViewNames;
import edu.illinois.cs.cogcomp.core.datastructures.textannotation.*;
import edu.illinois.cs.cogcomp.nlp.corpusreaders.PropbankReader;
import edu.illinois.cs.cogcomp.srl.caches.SentenceDBHandler;
import edu.illinois.cs.cogcomp.srl.data.Dataset;


import javax.xml.soap.Text;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

public class Main{

	static String[] allSectionsArray = Properties.getAllSections();
	static List<String> trainSections = Arrays.asList(Properties.getAllTrainSections());
	static List<String> testSections = Collections.singletonList(Properties.getTestSections());
	static List<String> trainDevSections = Arrays.asList(Properties.getTrainDevSections());
	static List<String> devSections = Arrays.asList(Properties.getDevSections());
	static String treebankHome = "/shared/corpora/corporaWeb/treebanks/eng/pennTreebank/treebank-3/parsed/mrg/wsj/";
	static String dataHome = "/shared/corpora/corporaWeb/treebanks/eng/propbank_1/data";
	static String goldView = ViewNames.SRL_VERB + "_GOLD";


	public static void main(String [] args) throws Exception {

		Iterator<TextAnnotation> data = new PropbankReader(treebankHome, dataHome, allSectionsArray,  goldView, true);
		//getDataStats(data);
		writeFileGoldSRL(data);

	}


	public static void writeFileGoldSRL(Iterator<TextAnnotation> data) throws Exception {
		String trainpath = "data/train.txt", devpath = "data/dev.txt", testpath = "data/test.txt";
		StringBuffer train = new StringBuffer();
		StringBuffer dev = new StringBuffer();
		StringBuffer test = new StringBuffer();


		while(data.hasNext()){
			TextAnnotation ta = data.next();
			if(ta.hasView(goldView)) {
				String id = ta.getId();
				String section = id.substring(id.indexOf('/') + 1, id.lastIndexOf('/'));
				if (trainSections.contains(section))
					train.append(getGoldSRL(ta));
				if (devSections.contains(section))
					dev.append(getGoldSRL(ta));
				if (testSections.contains(section))
					test.append(getGoldSRL(ta));

			}
		}

		BufferedWriter bwtrain = new BufferedWriter(new FileWriter(trainpath));
		bwtrain.write(train.toString());
		bwtrain.close();

		BufferedWriter bwdev = new BufferedWriter(new FileWriter(devpath));
		bwdev.write(dev.toString());
		bwdev.close();

		BufferedWriter bwtest = new BufferedWriter(new FileWriter(testpath));
		bwtest.write(test.toString());
		bwtest.close();

	}

	public static String getDataSetInfo(TextAnnotation ta){
		String id = ta.getId();
		String section = id.substring(id.indexOf('/') + 1, id.lastIndexOf('/'));
		String output = "NIL";
		if (trainSections.contains(section))
			output = "train";
		if (devSections.contains(section))
			output = "dev";
		if (testSections.contains(section))
			output = "test";

		return output;
	}

	public static void  getDataStats(Iterator<TextAnnotation> data) throws Exception {
		int train = 0, dev = 0, test = 0;
		while (data.hasNext()) {
			TextAnnotation ta = data.next();
			if (ta.hasView(goldView)) {
				if (getDataSetInfo(ta).equals("train"))
					train++;
				if (getDataSetInfo(ta).equals("dev"))
					dev++;
				if (getDataSetInfo(ta).equals("test"))
					test++;
			}
		}
		System.out.println("Train Data : " + train + " Dev Data : " + dev + " Test Data : " + test);
	}

	public static String getGoldSRL(TextAnnotation ta){
		StringBuffer output = new StringBuffer();
		String [] tokens = ta.getTokens();
		PredicateArgumentView span = (PredicateArgumentView) ta.getView(goldView);
		int num_predicates = span.getPredicates().size();



		List<Constituent> pa_cons = span.getConstituents();



		List<Constituent> cons = span.getConstituents();
		int c = 0, working_predicate = 0, encounter_p = -1;

		while(c < cons.size()) {
			if (cons.get(c).getLabel().equals("Predicate")) {
				if(working_predicate++ == ++encounter_p){
					String [] labels = new String[tokens.length];
					Arrays.fill(labels, "_NIL");
					int pred_token = cons.get(c).getStartSpan();
					labels[pred_token] = "_PRED";

					//if(++c < cons.size()) {
					while (++c < cons.size() && !cons.get(c).getLabel().equals("Predicate")) {
						int start = cons.get(c).getStartSpan();
						int end = cons.get(c).getEndSpan();
						String l = "_" + cons.get(c).getLabel();
						Arrays.fill(labels, start, end, l);
					}
					//}

					for(int tok = 0; tok < tokens.length; tok++)
						output.append(tokens[tok] + labels[tok] + " ");
					output.append("\n");
				}
			}
			else
				c++;
		}

		return output.toString();
		//return output;
	}
	
	public static void printPAStructure(TextAnnotation ta){
		PredicateArgumentView span = (PredicateArgumentView) ta.getView(goldView);
		if (ta.hasView(goldView)) {
			for(Constituent constituent : span.getConstituents())
				System.out.println(constituent + " " + constituent.getLabel() + " " + constituent.getStartSpan() + " " + constituent.getEndSpan());
		}

	}





}
