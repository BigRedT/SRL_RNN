#mvn clean
#mvn dependency:copy-dependencies
mvn compile

DEFAULT_PACKAGE="edu.illinois.cs.cogcomp"
PACKAGE=""
MAINCLASS="Main"
CP="./:./target/classes/:./target/dependency/*:./config/"
java -cp $CP $DEFAULT_PACKAGE.$MAINCLASS
