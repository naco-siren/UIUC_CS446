/**
 * Created by nacos on 2016/9/21.
 */
public class VectorUtil {

    public static double[] doAddition (final double[] a, final double[] b){
        assert(a.length == b.length);

        double[] sum = new double[a.length];
        for(int i = 0; i < a.length; i++){
            sum[i] = a[i] + b[i];
        }
        return sum;
    }

    public static double doInnerProduct(final double[] a, final double[] b){
        assert(a.length == b.length);

        double product = 0.0;
        for(int i = 0; i < a.length; i++){
            product += (a[i] * b[i]);
        }
        return product;
    }

    public static double predictLabel(final double[] weights, final double[] instance){
        double product = doInnerProduct(weights, instance);
        return product > 0? 1.0 : -1.0;
    }
}
