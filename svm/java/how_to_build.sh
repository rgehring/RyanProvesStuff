
# puts all dependencies on the classpath assuming you've installed the java matrix toolkit
# saves having to build a full maven project for these couple code examples requiring linear algebra

export CLASSPATH=${CLASSPATH}:./lib/*

javac kernels.java
javac svm_fit.java
java  SVMFit


