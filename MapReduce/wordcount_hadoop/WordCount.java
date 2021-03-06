import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import java.io.IOException;
import java.util.StringTokenizer;

public class WordCount {
    /***************************
    * MapReduceBase类:实现了Mapper和Reducer接口的基类（其中的方法只是实现接口，而未作任何事情） 
    * Mapper接口： 
    * WritableComparable接口：实现WritableComparable的类可以相互比较。所有被用作key的类应该实现此接口。 
    * Reporter 则可用于报告整个应用的运行进度，本例中未使用。  
    ***************************/
    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable>{
        /***************************
        * LongWritable, IntWritable, Text 均是 Hadoop 中实现的用于封装 Java 数据类型的类，这些类实现了WritableComparable接口， 
        * 都能够被串行化从而便于在分布式环境中进行数据交换，你可以将它们分别视为long,int,String 的替代品。 
        ***************************/ 
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();//Text 实现了BinaryComparable类可以作为key值
        /***************************
        * Mapper接口中的map方法： 
        * void map(K1 key, V1 value, OutputCollector<K2,V2> output, Reporter reporter) 
        * 映射一个单个的输入k/v对到一个中间的k/v对 
        * 输出对不需要和输入对是相同的类型，输入对可以映射到0个或多个输出对。 
        * OutputCollector接口：收集Mapper和Reducer输出的<k,v>对。 
        * OutputCollector接口的collect(k, v)方法:增加一个(k,v)对到output 
        ***************************/  
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            // 用StringTokenizer作为分词器，对value进行分词
            StringTokenizer itr = new StringTokenizer(value.toString());
            // 遍历分词后结果
            while (itr.hasMoreTokens()) {
                // 将String设置入Text类型word
                word.set(itr.nextToken());
                // 将(word,1)，即(Text,IntWritable)写入上下文context，供后续Reduce阶段使用
                context.write(word, one);
            }
        }
    }

    // IntSumReducer作为Reduce阶段，需要继承Reducer，并重写reduce()函数
    public static class IntSumReducer extends Reducer<Text,IntWritable,Text,IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            // 遍历map阶段输出结果中的values中每个val，累加至sum
            for (IntWritable val : values) {
                sum += val.get();
            }
            // 将sum设置入IntWritable类型result
            result.set(sum);
            // 通过上下文context的write()方法，输出结果(key, result)，即(Text,IntWritable)
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (otherArgs.length < 2) {
            System.err.println("Usage: wordcount <in> [<in>...] <out>");
            System.exit(2);
        }
        // 构造一个Job实例job，并命名为"word count"
        Job job = Job.getInstance(conf, "word count");
        // 设置jar
        job.setJarByClass(WordCount.class);         
        job.setMapperClass(TokenizerMapper.class);  // 为job设置Mapper类 
        job.setCombinerClass(IntSumReducer.class);  // 为job设置Combiner类  
        job.setReducerClass(IntSumReducer.class);   // 为job设置Reduce类 

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        for (int i = 0; i < otherArgs.length - 1; ++i) {
            FileInputFormat.addInputPath(job, new Path(otherArgs[i]));
        }
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[otherArgs.length - 1]));
        // 等待作业job运行完成并退出
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
