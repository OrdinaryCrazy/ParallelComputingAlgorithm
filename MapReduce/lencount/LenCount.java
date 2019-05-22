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

public class LenCount {
    public static class CounterMapper extends Mapper<Object, Text, Text, IntWritable>{
        private final static IntWritable one = new IntWritable(1);
        private Text word_len = new Text();
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            // 用StringTokenizer作为分词器，对value进行分词
            StringTokenizer itr = new StringTokenizer(value.toString());
            // 遍历分词后结果
            while (itr.hasMoreTokens()) {
                // 将String设置入Text类型word
                word_len.set(Integer.toString(itr.nextToken().length()));
                // 将(word,1)，即(Text,IntWritable)写入上下文context，供后续Reduce阶段使用
                context.write(word_len, one);
            }
        }
    }
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
            System.err.println("Usage: wordlen <in> [<in>...] <out>");
            System.exit(2);
        }
        // 构造一个Job实例job，并命名为"wordlen count"
        Job job = Job.getInstance(conf, "wordlen count");
        // 设置jar
        job.setJarByClass(LenCount.class);         
        job.setMapperClass(CounterMapper.class);    // 为job设置Mapper类 
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
