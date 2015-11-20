package edu.illinois.cs.cogcomp;

/**
 * Created by ngupta19 on 11/14/15.
 */
public class Properties {

    public static String[] getAllTrainSections() {
        return new String[] { "02", "03", "04", "05", "06", "07", "08", "09",
                "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
                "20", "21", "22" };
    }

    public static String getTestSections() {
        return "23";
    }

    public static String[] getAllSections() {
        return new String[] { "02", "03", "04", "05", "06", "07", "08", "09",
                "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
                "20", "21", "22", "24", "23" };

    }

    public static String[] getTrainDevSections() {
        return new String[] { "02", "03", "04", "05", "06", "07", "08",
                "09", "10", "11", "12", "13", "14", "15", "16", "17", "18",
                "19", "20", "21", "22", "24" };
    }

    public static String[] getDevSections() {
        return new String[] { "24" };
    }
}
