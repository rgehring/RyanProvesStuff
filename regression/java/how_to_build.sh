
# puts all dependencies on the classpath assuming you've installed the java matrix toolkit
# saves having to build a full maven project for these couple code examples requiring linear algebra

export CLASSPATH=${CLASSPATH}:./lib/*

javac regression_tools.java
javac run_regression.java
java RunRegression -i /data/bears.csv -y AGE


