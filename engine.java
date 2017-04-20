/**
 * Created by edward on 18/04/2017.
 */
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;

import java.util.ArrayList;
import java.util.List;
import org.apache.spark.api.java.*;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import scala.Tuple2;
import scala.util.parsing.combinator.testing.Str;

public class engine {
    public static void main(String[] args) throws Exception{

        // mode and app name setting
        String localmode = "local[1]";
        String clustermode = "yarn-cluster";
        String appname ="simpleapp";

        //SparkConf sparkConf = new SparkConf().setMaster("yarn-cluster").setAppName("JavaSparkPi");
        SparkConf sparkConf = ModeChoose(localmode, appname);
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);

        // Load the data
        String path = "C:\\spark-2.1.0-bin-hadoop2.7\\bin\\movie\\ratings.csv";
        JavaRDD<String> data = jsc.textFile(path);
        //drop timestamp column in data
        data = data.filter(new Function<String, Boolean>() {
            public Boolean call(String s) throws Exception {
                if (s.charAt(0)!= 'u')
                    return true;
                else return false ;

            }
        });
        data = data.map(new Function<String, String>() {
            public String call(String s) throws Exception {
                String[] temp = s.split(",");
                String r = temp[0]+','+temp[1]+','+ temp[2];
                return r;
            }
        });

        // Split initial RDD data into two parts [60% training data, 40% testing data].
        JavaRDD<String>[] splits =
                data.randomSplit(new double[]{0.9, 0.1}, 11L);
        // *****training data
        JavaRDD<String > training = splits[0];

        // *****test data
        JavaRDD<String> test = splits[1];

        //turn JavaRDD<String> to JavaRDD<Rating> then RDD<Rating>, the standard type for ALS
        JavaRDD<Rating> ratings = training.map(
                new Function<String, Rating>() {
                    public Rating call(String s) {
                        String[] sarray = s.split(",");
                        return new Rating(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]),
                                Double.parseDouble(sarray[2]));
                    }
                }
        );
        // transform test data into Rating format
        JavaRDD<Rating> testRatings = test.map(
                new Function<String, Rating>() {
                    public Rating call(String s) {
                        String[] sarray = s.split(",");
                        return new Rating(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]),
                                Double.parseDouble(sarray[2]));
                    }
                }
        );




        // Build the recommendation model using ALS
        int rank = 10;
        int numIterations = 10;
        double v =0.01;

        int[] rank_iter= modelParaChoose(rank,numIterations,v,ratings,testRatings);

        MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratings), rank_iter[0], rank_iter[1], v);

        //save model
        model.save(jsc.sc(), "C:\\spark-2.1.0-bin-hadoop2.7\\bin\\model");
        /*
        // get user and products from test data
        JavaRDD<Tuple2<Object, Object>> userProducts = testRatings.map(
                new Function<Rating, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(Rating r) {
                        return new Tuple2<Object, Object>(r.user(), r.product());
                    }
                }
        );
        // get predictions
        JavaPairRDD<Tuple2<Integer, Integer>, Double> predictions = JavaPairRDD.fromJavaRDD(
                model.predict(JavaRDD.toRDD(userProducts)).toJavaRDD().map(
                        new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
                            public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r){
                                return new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating());
                            }
                        }
                ));

        // put test data ratings and predictions together
        JavaRDD<Tuple2<Double, Double>> ratesAndPreds =
                JavaPairRDD.fromJavaRDD(testRatings.map(
                        new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
                            public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r){
                                return new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating());
                            }
                        }
                )).join(predictions).values();
        // calculate Mean Squared error
        double MSE = JavaDoubleRDD.fromRDD(ratesAndPreds.map(
                new Function<Tuple2<Double, Double>, Object>() {
                    public Object call(Tuple2<Double, Double> pair) {
                        Double err = pair._1() - pair._2();
                        return err * err;
                    }
                }
        ).rdd()).mean();
        System.out.println("Mean Squared Error = " + MSE);
        */

        /*
        //show selected rdd element

        List<String> ele1= test.take(1);
        System.out.println(ele1.toString());
        */
        jsc.stop();

    }

    public static SparkConf ModeChoose( String mode, String appname)
    {
        SparkConf s1 = new SparkConf().setMaster( mode ).setAppName(appname) ;
        return  s1;
    }

    public static int[] modelParaChoose ( int rank, int iteration, double v, JavaRDD<Rating> ratings, JavaRDD<Rating> testRatings )
    {
        double tempMSE =0;
        int[] paraRec = new int[2];

        for (int i=1; i<rank+1; i++){
            for (int j=1; j<iteration+1; j++)
            {
                MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratings), i, j, v);

                // get user and products from test data
                JavaRDD<Tuple2<Object, Object>> userProducts = testRatings.map(
                        new Function<Rating, Tuple2<Object, Object>>() {
                            public Tuple2<Object, Object> call(Rating r) {
                                return new Tuple2<Object, Object>(r.user(), r.product());
                            }
                        }
                );
                // get predictions
                JavaPairRDD<Tuple2<Integer, Integer>, Double> predictions = JavaPairRDD.fromJavaRDD(
                        model.predict(JavaRDD.toRDD(userProducts)).toJavaRDD().map(
                                new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
                                    public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r){
                                        return new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating());
                                    }
                                }
                        ));

                // put test data ratings and predictions together
                JavaRDD<Tuple2<Double, Double>> ratesAndPreds =
                        JavaPairRDD.fromJavaRDD(testRatings.map(
                                new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
                                    public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r){
                                        return new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating());
                                    }
                                }
                        )).join(predictions).values();
                // calculate Mean Squared error
                double MSE = JavaDoubleRDD.fromRDD(ratesAndPreds.map(
                        new Function<Tuple2<Double, Double>, Object>() {
                            public Object call(Tuple2<Double, Double> pair) {
                                Double err = pair._1() - pair._2();
                                return err * err;
                            }
                        }
                ).rdd()).mean();
                System.out.println("Mean Squared Error = " + MSE);

                if ( i==1&&j==1 )
                {
                    tempMSE = MSE;
                    paraRec[0] = i;
                    paraRec[1] = j;
                }
                else {
                    if(MSE < tempMSE)
                    {
                        tempMSE =MSE;
                        paraRec[0] = i;
                        paraRec[1] = j;
                    }
                }

            }
        }

        System.out.println("min Mean Squared Error = " + tempMSE);
        System.out.println("Corresponding rank = " + paraRec[0]);
        System.out.println("Corresponding numiteration = " + paraRec[1]);
        return  paraRec;
    }


}
