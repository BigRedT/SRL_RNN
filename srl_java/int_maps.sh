#mvn clean
#mvn dependency:copy-dependencies
mvn compile

DEFAULT_PACKAGE="edu.illinois.cs.cogcomp"
PACKAGE="cureData"
MAINCLASS="Main"
CP="./:./target/classes/:./target/dependency/*:./config/"
java -cp $CP $DEFAULT_PACKAGE.$PACKAGE.$MAINCLASS
