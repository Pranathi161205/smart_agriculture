import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.VectorWritable;

public class VectorSequenceConverter {

    public static void main(String[] args) throws IOException {

        String inputFile = args[0];
        String outputFile = args[1];

        Configuration conf = new Configuration();

        Path outputPath = new Path(outputFile);

        SequenceFile.Writer writer = SequenceFile.createWriter(
            conf,
            SequenceFile.Writer.file(outputPath),
            SequenceFile.Writer.keyClass(Text.class),
            SequenceFile.Writer.valueClass(VectorWritable.class)
        );

        try (BufferedReader br = new BufferedReader(
                new FileReader(inputFile))) {

            String line;

            while ((line = br.readLine()) != null) {

                String[] parts = line.split("\t");

                String imageId = parts[0];

                double[] values = new double[6];

                for (int i = 0; i < 6; i++) {
                    values[i] = Double.parseDouble(parts[i + 1]);
                }

                DenseVector vector = new DenseVector(values);

                writer.append(
                    new Text(imageId),
                    new VectorWritable(vector)
                );
            }
        }

        writer.close();

        System.out.println("SequenceFile created successfully!");
    }
}
