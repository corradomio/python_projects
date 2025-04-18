@echo off
CLASSPATH=./lib/ui.jar
CLASSPATH=$CLASSPATH:./lib/core.jar
CLASSPATH=$CLASSPATH:./lib/jext-concurrent.jar
CLASSPATH=$CLASSPATH:./lib/json-simple-1.1.1.jar
CLASSPATH=$CLASSPATH:./lib/plotly.java-main.jar
CLASSPATH=$CLASSPATH:./lib/postgresql-42.2.2.jar
CLASSPATH=$CLASSPATH:./lib/Soptimtools-3.7.jar
CLASSPATH=$CLASSPATH:./lib/Unfolding.jar
CLASSPATH=$CLASSPATH:./lib/ooxml-lib/commons-compress-1.20.jar
CLASSPATH=$CLASSPATH:./lib/ooxml-lib/curvesapi-1.06.jar
CLASSPATH=$CLASSPATH:./lib/ooxml-lib/xmlbeans-4.0.0.jar
CLASSPATH=$CLASSPATH:./lib/opencsv/commons-beanutils-1.9.4.jar
CLASSPATH=$CLASSPATH:./lib/opencsv/commons-collections-3.2.2.jar
CLASSPATH=$CLASSPATH:./lib/opencsv/commons-collections4-4.4.jar
CLASSPATH=$CLASSPATH:./lib/opencsv/commons-lang3-3.12.0.jar
CLASSPATH=$CLASSPATH:./lib/opencsv/commons-logging-1.2.jar
CLASSPATH=$CLASSPATH:./lib/opencsv/commons-text-1.10.0.jar
CLASSPATH=$CLASSPATH:./lib/opencsv/opencsv-5.7.1.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/poi-5.0.0.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/poi-examples-5.0.0.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/poi-excelant-5.0.0.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/poi-integration-5.0.0.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/poi-ooxml-5.0.0.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/poi-ooxml-full-5.0.0.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/poi-ooxml-lite-5.0.0.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/poi-scratchpad-5.0.0.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/auxiliary/batik-all-1.13.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/auxiliary/bcpkix-jdk15on-1.68.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/auxiliary/bcprov-jdk15on-1.68.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/auxiliary/fontbox-2.0.22.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/auxiliary/graphics2d-0.30.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/auxiliary/pdfbox-2.0.22.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/auxiliary/xml-apis-ext-1.3.04.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/auxiliary/xmlgraphics-commons-2.4.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/auxiliary/xmlsec-2.2.1.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/lib/commons-codec-1.15.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/lib/commons-collections4-4.4.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/lib/commons-math3-3.6.1.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/lib/SparseBitSet-1.2.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/ooxml-lib/commons-compress-1.20.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/ooxml-lib/curvesapi-1.06.jar
CLASSPATH=$CLASSPATH:./lib/poi-5.0.0/ooxml-lib/xmlbeans-4.0.0.jar
CLASSPATH=$CLASSPATH:./lib/tablesaw-plot/checker-qual-3.5.0.jar
CLASSPATH=$CLASSPATH:./lib/tablesaw-plot/classgraph-4.8.60.jar
CLASSPATH=$CLASSPATH:./lib/tablesaw-plot/commons-math3-3.6.1.jar
CLASSPATH=$CLASSPATH:./lib/tablesaw-plot/error_prone_annotations-2.3.4.jar
CLASSPATH=$CLASSPATH:./lib/tablesaw-plot/failureaccess-1.0.1.jar
CLASSPATH=$CLASSPATH:./lib/tablesaw-plot/fastutil-8.3.0.jar
CLASSPATH=$CLASSPATH:./lib/tablesaw-plot/guava-30.0-jre.jar
CLASSPATH=$CLASSPATH:./lib/tablesaw-plot/icu4j-65.1.jar
CLASSPATH=$CLASSPATH:./lib/tablesaw-plot/j2objc-annotations-1.3.jar
CLASSPATH=$CLASSPATH:./lib/tablesaw-plot/jackson-annotations-2.13.2.jar
CLASSPATH=$CLASSPATH:./lib/tablesaw-plot/jackson-core-2.13.2.jar
CLASSPATH=$CLASSPATH:./lib/tablesaw-plot/jackson-databind-2.13.2.1.jar
CLASSPATH=$CLASSPATH:./lib/tablesaw-plot/jsr305-3.0.2.jar
CLASSPATH=$CLASSPATH:./lib/tablesaw-plot/pebble-3.1.2.jar
CLASSPATH=$CLASSPATH:./lib/tablesaw-plot/RoaringBitmap-0.9.25.jar
CLASSPATH=$CLASSPATH:./lib/tablesaw-plot/shims-0.9.25.jar
CLASSPATH=$CLASSPATH:./lib/tablesaw-plot/slf4j-api-1.7.25.jar
CLASSPATH=$CLASSPATH:./lib/tablesaw-plot/tablesaw-core-0.43.1.jar
CLASSPATH=$CLASSPATH:./lib/tablesaw-plot/unbescape-1.1.6.RELEASE.jar
CLASSPATH=$CLASSPATH:./lib/tablesaw-plot/univocity-parsers-2.8.4.jar

java -cp $CLASSPATH test.aa.MainTestDriveGA

