/**
 * Christof Seiler, 
 * Department of Statistics, Stanford University
 */

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkExtractImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkFlipImageFilter.h>
#include <itkShrinkImageFilter.h>
#include <itkExpandImageFilter.h>
#include <itkDirectory.h>
#include <itkRegularExpressionSeriesFileNames.h>
#include <itkFileTools.h>
#include <itkImageRegionConstIteratorWithOnlyIndex.h>
#include <itkImageDuplicator.h>

#include <vnl_sd_matrix_tools.h>

#include <vnl/vnl_trace.h>
#include <vnl/vnl_inverse.h>
#include <vnl/algo/vnl_cholesky.h>
#include <vnl/vnl_sparse_matrix.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/math/special_functions/gamma.hpp>

#include <iostream>

const unsigned int Dimension = 3;

typedef float VectorValueType;
typedef itk::Vector< VectorValueType,Dimension > VectorType;
typedef itk::Image< VectorType,Dimension > VectorFieldType;

//typedef signed short PixelType;
typedef float PixelType;
typedef itk::Image< PixelType, Dimension > ImageType;

typedef unsigned int LabelPixelType;
typedef itk::Image< LabelPixelType, Dimension > LabelImageType;

const LabelImageType::PixelType NoOfNeighborsWithinDistance1 = Dimension*2;

LabelImageType::IndexType offset0 = {{0,0,-1}};
LabelImageType::IndexType offset1 = {{0,0,1}};
LabelImageType::IndexType offset2 = {{0,-1,0}};
LabelImageType::IndexType offset3 = {{0,1,0}};
LabelImageType::IndexType offset4 = {{-1,0,0}};
LabelImageType::IndexType offset5 = {{1,0,0}};
LabelImageType::IndexType center = {{0,0,0}};
LabelImageType::IndexType offset[NoOfNeighborsWithinDistance1+1] = {offset0,offset1,offset2,offset3,offset4,offset5,center};

vnl_matrix< double > KroneckerProduct(vnl_matrix< double > A, vnl_matrix< double > B) {
    
    // C = A (KroneckerProduct) B
    vnl_matrix< double > C(A.rows()*B.rows(),A.cols()*B.cols());
    
    for(unsigned int i = 0; i < A.rows(); ++i) {
        for(unsigned int j = 0; j < A.cols(); ++j) {
            
            for(unsigned int m = 0; m < B.rows(); ++m) {
                for(unsigned int n = 0; n < B.cols(); ++n) {
                    C(i*B.rows()+m,j*B.cols()+n) = A(i,j)*B(m,n);
                }
            }
            
        }
    }
    
    return C;
    
}

double computeLogMarginalLikelihood(ImageType::Pointer templateImage, std::vector< LabelPixelType >& partition, std::vector< VectorFieldType::Pointer > velocityFields, const double variance, const vnl_matrix< double > priorPrecision, const LabelImageType::PixelType partitionLabel) {
    
    vnl_matrix< double > Id3(Dimension,Dimension);
    Id3.set_identity();
    ImageType::SpacingType spacing = templateImage->GetSpacing();
    
    itk::ImageRegionConstIteratorWithOnlyIndex< ImageType > templateIter(templateImage, templateImage->GetLargestPossibleRegion());
    
    double logMarginalLikelihood = 0;
    for(unsigned int i = 0; i < velocityFields.size(); ++i) {
        
        unsigned int noOfVoxels = 0;
        for(unsigned int j = 0; j < partition.size(); ++j) {
            if( partition[j] == partitionLabel )
                ++noOfVoxels;
        }
        
        vnl_matrix< double > X(Dimension+1,noOfVoxels);
        vnl_matrix< double > vectV(Dimension*noOfVoxels,1);

        unsigned int voxelId = 0, partitionId = 0;
        for(templateIter.GoToBegin(); !templateIter.IsAtEnd(); ++templateIter) {
            if( partition[partitionId] == partitionLabel ) {
                ImageType::IndexType index = templateIter.GetIndex();
                VectorFieldType::PixelType value = velocityFields[i]->GetPixel(index);
                
                for(unsigned int j = 0; j < Dimension; ++j) {
                    X(j,voxelId) = index[j] * spacing[j];
                    vectV(voxelId*Dimension+j,0) = value[j];
                }
                X(Dimension,voxelId) = 1;
                ++voxelId;
            }
            ++partitionId;
        }

//            std::cout << "X = \n" << X << std::endl;
//            std::cout << "vectV = \n" << vectV << std::endl;
        
        vnl_matrix< double> Phi = KroneckerProduct(X.transpose(), Id3);
//            std::cout << "Phi = \n" << Phi << std::endl;
        
        vnl_matrix< double > GammaCheck = 1/variance*Phi.transpose()*Phi + priorPrecision;
//            std::cout << "GammaCheck = \n" << GammaCheck << std::endl;
        double term1 = 0.5*std::log(vnl_determinant(variance*priorPrecision));
        double term2 = Dimension*noOfVoxels/2.0*std::log(2.0*itk::Math::pi*variance);
        double term3 = 0.5*std::log(vnl_determinant(variance*GammaCheck));
        double normalizingConstant = term1 - term2 - term3;
        vnl_cholesky choleskyVarGammaCheck(variance*GammaCheck);
//            std::cout << "choleskyVarGammaCheck.inverse() = \n" << choleskyVarGammaCheck.inverse() << std::endl;
        logMarginalLikelihood += normalizingConstant + 1.0/(2.0*variance) * (vectV.transpose() * Phi * choleskyVarGammaCheck.inverse() * Phi.transpose() * vectV)(0,0);
        
    }
    
    return logMarginalLikelihood;
}

double computeLogMarginalLikelihood(ImageType::Pointer templateImage, std::vector< LabelPixelType >& partition, std::vector< VectorFieldType::Pointer > velocityFields, const double variance, const vnl_matrix< double > priorPrecision) {
    
    unsigned int noOfPartitions = *std::max_element(partition.begin(), partition.end());
    
    double logMarginalLikelihood = 0;
    for(unsigned int k = 1; k <= noOfPartitions; ++k)
        logMarginalLikelihood += computeLogMarginalLikelihood(templateImage, partition, velocityFields, variance, priorPrecision, k);
    
    return logMarginalLikelihood;
    
}

std::string writePartitionStatistics(LabelImageType::Pointer partitionImage) {
    
    typedef itk::MinimumMaximumImageCalculator< LabelImageType >  MinimumMaximumImageCalculatorType;
    MinimumMaximumImageCalculatorType::Pointer minMaxFilter = MinimumMaximumImageCalculatorType::New();
    minMaxFilter->SetImage(partitionImage);
    minMaxFilter->Compute();
    unsigned int noOfPartitions = minMaxFilter->GetMaximum();
    std::cout << "noOfPartitions = " << noOfPartitions << std::endl;
    
    std::vector< unsigned int > paritionMass(noOfPartitions+1);
    for(unsigned int i = 0; i < paritionMass.size(); ++i)
        paritionMass[i] = 0;
    typedef itk::ImageRegionConstIterator< LabelImageType > ImageRegionConstIteratorType;
    ImageRegionConstIteratorType iterMass(partitionImage,partitionImage->GetLargestPossibleRegion());
    for(iterMass.GoToBegin(); !iterMass.IsAtEnd(); ++iterMass)
        ++paritionMass[iterMass.Get()];
    
    std::ostringstream osstr;
    for(unsigned int i = 0; i < paritionMass.size(); ++i)
        osstr << paritionMass[i] << " ";
    osstr << std::endl;
    
    return osstr.str();
    
}

void writePartition(ImageType::Pointer templateImage, std::vector< LabelPixelType >& partition, const unsigned int axis, const unsigned int slice, const unsigned int visualExpandFactor, const unsigned int step, const unsigned int shrinkFactor) {
    
    LabelImageType::Pointer partitionImage = LabelImageType::New();
    partitionImage->CopyInformation(templateImage);
    partitionImage->SetRegions(templateImage->GetLargestPossibleRegion());
    partitionImage->Allocate();
    
    itk::ImageRegionIterator< LabelImageType > partitionIter(partitionImage, partitionImage->GetLargestPossibleRegion());
    unsigned id = 0;
    for(partitionIter.GoToBegin(); !partitionIter.IsAtEnd(); ++partitionIter) {
        partitionIter.Set(partition[id]);
        ++id;
    }
    
    itk::ImageFileWriter< LabelImageType >::Pointer labelWriter = itk::ImageFileWriter< LabelImageType >::New();
    labelWriter->SetInput(partitionImage);
    std::ostringstream osstr;
    osstr << "ParitionImage_Step" << step << ".mha";
    labelWriter->SetFileName(osstr.str().c_str());
    labelWriter->Update();
    
    typedef itk::ImageDuplicator< LabelImageType > DuplicatorType;
    DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage(partitionImage);
    duplicator->Update();
    
    const unsigned int SliceDimension = 2;
    typedef itk::Image< LabelPixelType,SliceDimension > OutputImageType;
    typedef itk::Image< unsigned char,SliceDimension > ImageSliceType;
    
    ImageType::RegionType inputRegion = duplicator->GetOutput()->GetLargestPossibleRegion();
    ImageType::SizeType size = inputRegion.GetSize();
    size[axis] = 0;

    ImageType::IndexType start = inputRegion.GetIndex();
    start[axis] = slice/shrinkFactor;
//    start[axis] = slice;

    ImageType::RegionType desiredRegion;
    desiredRegion.SetSize(size);
    desiredRegion.SetIndex(start);

    typedef itk::ExtractImageFilter< LabelImageType,OutputImageType > ExtractImageFilterType;
    ExtractImageFilterType::Pointer extractFilter = ExtractImageFilterType::New();
    extractFilter->SetInput(duplicator->GetOutput());
    extractFilter->SetExtractionRegion(desiredRegion);
    extractFilter->SetDirectionCollapseToIdentity();

    typedef itk::RescaleIntensityImageFilter< OutputImageType,ImageSliceType > RescaleImageFilterType;
    RescaleImageFilterType::Pointer rescaleFilter = RescaleImageFilterType::New();
    rescaleFilter->SetOutputMinimum(itk::NumericTraits<unsigned char>::NonpositiveMin());
    rescaleFilter->SetOutputMaximum(itk::NumericTraits<unsigned char>::max());
    rescaleFilter->SetInput(extractFilter->GetOutput());

    typedef itk::NearestNeighborInterpolateImageFunction< ImageSliceType > SliceInterpolatorType;
    SliceInterpolatorType::Pointer sliceInterpolator = SliceInterpolatorType::New();
    typedef itk::ExpandImageFilter< ImageSliceType,ImageSliceType > ExpandImageFilterType;
    ExpandImageFilterType::Pointer expandFilter = ExpandImageFilterType::New();
    expandFilter->SetInput(rescaleFilter->GetOutput());
    expandFilter->SetInterpolator(sliceInterpolator);
    expandFilter->SetExpandFactors(visualExpandFactor);

    typedef itk::FlipImageFilter< ImageSliceType > FlipImageFilterType;
    FlipImageFilterType::Pointer flipFilter = FlipImageFilterType::New();
    flipFilter->SetInput(expandFilter->GetOutput());
    FlipImageFilterType::FlipAxesArrayType flipAxes;
    flipAxes[0] = false;
    flipAxes[1] = true;
    flipFilter->SetFlipAxes(flipAxes);

    typedef itk::ImageFileWriter< ImageSliceType > WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetInput(flipFilter->GetOutput());
    std::ostringstream osstrPNG;
    osstrPNG << "ParitionImage_Step" << step << ".png";
    writer->SetFileName(osstrPNG.str().c_str());
    writer->Update();

}

const double pi3over2 = 5.5683279968317078452848179821188357020136243902832439;
double Gamma3(double a) {
    return pi3over2 * boost::math::tgamma(a) * boost::math::tgamma(a-0.5) * boost::math::tgamma(a-1);
}

// 1.5*std::log(itk::Math::pi);
const double logpi = 1.1447298858494001741434273513530587116472948129153115;
double LogGamma3(double a) {
    return 1.5*logpi + boost::math::lgamma(a) + boost::math::lgamma(a-0.5) + boost::math::lgamma(a-1);
}

VectorFieldType::Pointer createField(ImageType::Pointer templateImage, vnl_matrix< double > logA) {
    
    VectorFieldType::Pointer velocityField = VectorFieldType::New();
    velocityField->SetRegions(templateImage->GetLargestPossibleRegion());
    velocityField->Allocate();

    itk::ImageRegionIteratorWithIndex< VectorFieldType > fieldIter(velocityField, velocityField->GetLargestPossibleRegion());
    for(fieldIter.GoToBegin(); !fieldIter.IsAtEnd(); ++fieldIter) {
        VectorFieldType::IndexType index = fieldIter.GetIndex();
        vnl_matrix< double > xHom(Dimension+1,1);
        xHom(0,0) = index[0];
        xHom(1,0) = index[1];
        xHom(2,0) = index[2];
        xHom(Dimension,0) = 1;
        
        vnl_matrix< double > velocity = logA.extract(3,4) * xHom;
        VectorFieldType::PixelType value;
        value[0] = velocity(0,0);
        value[1] = velocity(1,0);
        value[2] = velocity(2,0);
        fieldIter.Set(value);
    }
    
    return velocityField;

}

void addNoise(VectorFieldType::Pointer velocityField, double sd) {
    
//    boost::variate_generator<boost::mt19937, boost::normal_distribution<> > rnormal(boost::mt19937(time(0)), boost::normal_distribution<>(0,sd));
    boost::variate_generator<boost::mt19937, boost::normal_distribution<> > rnormal(boost::mt19937(), boost::normal_distribution<>(0,sd));
    
    itk::ImageRegionIteratorWithIndex< VectorFieldType > fieldIter(velocityField, velocityField->GetLargestPossibleRegion());
    for(fieldIter.GoToBegin(); !fieldIter.IsAtEnd(); ++fieldIter) {
        VectorFieldType::PixelType value = fieldIter.Get();
        for(unsigned int i = 0; i < Dimension; ++i) {
            double noise = rnormal();
//            std::cout << noise << std::endl;
            value[i] += noise;
        }
        fieldIter.Set(value);
    }
}

unsigned int indexToId(ImageType::IndexType index, ImageType::SizeType size) {
    return index[0] + index[1]*size[0] + index[2]*size[0]*size[1];
}

ImageType::IndexType idToIndex(unsigned int id, ImageType::SizeType size) {
    ImageType::IndexType index;
    
    index[2] = floor(id/(size[0]*size[1]));
    unsigned int intermediate = index[2]*size[0]*size[1];
    index[1] = floor((id-intermediate)/size[0]);
    index[0] = id - (index[1]*size[0] + intermediate);
    
    return index;
}

void colorNodes(unsigned int id, vnl_sparse_matrix< int >& linkMatrix, LabelPixelType& label, std::vector< LabelPixelType >& partition, bool firstIteration) {
        
    if(partition[id] == 0) {
        // faster version
        if(firstIteration) {
            ++label;
            firstIteration = false;
        }
        partition[id] = label;
        vnl_sparse_matrix< int >::row pairs = linkMatrix.get_row(id);
        for(unsigned int j = 0; j < pairs.size(); ++j)
            colorNodes(pairs[j].first, linkMatrix, label, partition, firstIteration);
    }
}

void createPartition(std::vector< unsigned int >& links, std::vector< LabelPixelType >& partition) {
    
    for(unsigned int i = 0; i < partition.size(); ++i)
        partition[i] = 0;
    
    vnl_sparse_matrix< int > linkMatrix(links.size(),links.size());
    for(unsigned int i = 0; i < links.size(); ++i) {
        linkMatrix.put(i, links[i], 1);
        linkMatrix.put(links[i], i, 1);
    }
    
    LabelPixelType label = 0;
    for(unsigned int i = 0; i < partition.size(); ++i) {
        
//        // slow way of counting number of labels
//        unsigned int noOfPartitions = *std::max_element(partition.begin(), partition.end());
//        LabelPixelType label = noOfPartitions+1;
        
        // fast way by keeping track of a global count and the number of iteration taken at each step
        bool firstIteration = true;
        colorNodes(i, linkMatrix, label, partition, firstIteration);
    }
    
}

void testCase(std::string templateImageName, std::string vectorFieldDirName, unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ, double variance, double noise, vnl_matrix< double > priorPrecision) {
    
    // create template image
    ImageType::Pointer templateImage = ImageType::New();
    
    ImageType::RegionType region;
    ImageType::IndexType start;
    start[0] = 0;
    start[1] = 0;
    start[2] = 0;
    
    ImageType::SizeType size;
    size[0] = sizeX;
    size[1] = sizeY;
    size[2] = sizeZ;
    
    region.SetSize(size);
    region.SetIndex(start);
    
    templateImage->SetRegions(region);
    templateImage->Allocate();
    templateImage->FillBuffer(0);
    
    itk::ImageFileWriter< ImageType >::Pointer imageWriter = itk::ImageFileWriter< ImageType >::New();
    imageWriter->SetInput(templateImage);
    imageWriter->SetFileName(templateImageName);
    imageWriter->Update();
    
    // velocity field 1
    vnl_matrix< double > Ry(Dimension+1,Dimension+1);
    double angle = itk::Math::pi/9;
    Ry.set_identity();
    Ry(0,0) = cos(angle);
    Ry(2,0) = -sin(angle);
    Ry(0,2) = sin(angle);
    Ry(2,2) = cos(angle);
    std::cout << "Ry = \n" << Ry << std::endl;
    
    vnl_matrix< double > Rz(Dimension+1,Dimension+1);
    double angleZ = itk::Math::pi/18;
    Rz.set_identity();
    Rz(0,0) = cos(angleZ);
    Rz(1,0) = sin(angleZ);
    Rz(0,1) = -sin(angleZ);
    Rz(1,1) = cos(angleZ);
    std::cout << "Rz = \n" << Rz << std::endl;
    
    vnl_matrix< double > C1(Dimension+1,Dimension+1);
    C1.set_identity();
    C1(0,3) = 0.5*(sizeX-1);
    C1(1,3) = 0.5*(sizeY-1);
    C1(2,3) = 1.0/8.0*(sizeZ-1);
    std::cout << "C1 = \n" << C1 << std::endl;
    vnl_matrix< double > A1 = C1 * Rz*Ry * vnl_inverse(C1);
    vnl_matrix< double > logA1 = sdtools::GetLogarithm(A1);
    std::cout << "A1 = \n" << A1 << std::endl;
    
    VectorFieldType::Pointer velocityField1 = createField(templateImage, logA1);
    
    // velocity field 2
    vnl_matrix< double > T(Dimension+1,Dimension+1);
    T.set_identity();
    T(0,3) = -1.0;
    T(1,3) = 1.0;
//    T(2,3) = 0.25;
    
    vnl_matrix< double > logT = sdtools::GetLogarithm(T);
    std::cout << "T = \n" << T << std::endl;
    
    VectorFieldType::Pointer velocityField2 = createField(templateImage, logT);

    // velocity field 3
    vnl_matrix< double > C3(Dimension+1,Dimension+1);
    C3.set_identity();
    C3(0,3) = 0.5*(sizeX-1);
    C3(1,3) = 0.5*(sizeY-1);
    C3(2,3) = 5.0/8.0*(sizeZ-1);
    std::cout << "C3 = \n" << C3 << std::endl;
        
    vnl_matrix< double > A3 = C3 * Rz*vnl_inverse(Ry) * vnl_inverse(C3);
    vnl_matrix< double > logA3 = sdtools::GetLogarithm(A3);
    std::cout << "A3 = \n" << A3 << std::endl;
    
    VectorFieldType::Pointer velocityField3 = createField(templateImage, logA3);
    
    // velocity field 4
    vnl_matrix< double > S(Dimension+1,Dimension+1);
    S.set_identity();
    S(0,0) = 0.7;
    S(1,1) = 1.2;
    S(2,2) = 1.3;
    std::cout << "S = \n" << S << std::endl;
    
    vnl_matrix< double > C4(Dimension+1,Dimension+1);
    C4.set_identity();
    C4(0,3) = 0.5*(sizeX-1);
    C4(1,3) = 0.5*(sizeY-1);
    C4(2,3) = 7.0/8.0*(sizeZ-1);

    std::cout << "C4 = \n" << C4 << std::endl;
    vnl_matrix< double > A4 = C4 * S * vnl_inverse(C4);
    vnl_matrix< double > logA4 = sdtools::GetLogarithm(A4);
    std::cout << "A4 = \n" << A4 << std::endl;
    
    VectorFieldType::Pointer velocityField4 = createField(templateImage, logA4);
    
    // combine velocity fields
    VectorFieldType::Pointer velocityFieldCombined = VectorFieldType::New();
    velocityFieldCombined->SetRegions(templateImage->GetLargestPossibleRegion());
    velocityFieldCombined->Allocate();
    
    itk::ImageRegionIteratorWithIndex< VectorFieldType > fieldIter(velocityFieldCombined, velocityFieldCombined->GetLargestPossibleRegion());
    unsigned int noOfVoxels = sizeX*sizeY*sizeZ;
    unsigned int quarter = ceil(0.25*noOfVoxels);
    unsigned int half = ceil(0.5*noOfVoxels);
    unsigned int threeQuarters = ceil(0.75*noOfVoxels);
    unsigned int count = 0;
    for(fieldIter.GoToBegin(); !fieldIter.IsAtEnd(); ++fieldIter) {
        VectorFieldType::IndexType index = fieldIter.GetIndex();
        if(count < quarter)
            fieldIter.Set(velocityField1->GetPixel(index));
        else if(count < half)
            fieldIter.Set(velocityField2->GetPixel(index));
        else if(count < threeQuarters)
            fieldIter.Set(velocityField3->GetPixel(index));
        else
            fieldIter.Set(velocityField4->GetPixel(index));
        ++count;
    }
    
    addNoise(velocityFieldCombined, sqrt(noise));
    
    itk::ImageFileWriter< VectorFieldType >::Pointer fieldWriter = itk::ImageFileWriter< VectorFieldType >::New();
    std::ostringstream fieldFilename;
    fieldFilename << vectorFieldDirName << "/TestVectorField.mha";
    fieldWriter->SetFileName(fieldFilename.str());
    fieldWriter->SetInput(velocityFieldCombined);
    fieldWriter->Update();
    
    // test marginal likelihood
    std::vector< LabelPixelType > partition(noOfVoxels);
    for(unsigned int i = 0; i < partition.size(); ++i) {
        if(i < half)
            partition[i] = 1;
        else
            partition[i] = 2;
    }
    
    std::vector< VectorFieldType::Pointer > vectorFields(1);
    vectorFields[0] = velocityFieldCombined;
    double marginalTwoParts = computeLogMarginalLikelihood(templateImage, partition, vectorFields, variance, priorPrecision);
    std::cout << "marginal two part = " << marginalTwoParts << std::endl;
    
    for(unsigned int i = 0; i < partition.size(); ++i)
        partition[i] = 1;
    double marginalOnePart = computeLogMarginalLikelihood(templateImage, partition, vectorFields, variance, priorPrecision);
    std::cout << "marginal one parts = " << marginalOnePart << std::endl;
    
    double bayesFactor = std::exp(marginalTwoParts - marginalOnePart);
    std::cout << "bayes factor, std::exp(marginalTwoParts - marginalOnePart) = " << bayesFactor << std::endl;
    
    for(unsigned int i = 0; i < partition.size(); ++i) {
        if(i < quarter)
            partition[i] = 1;
        else if(i < half)
            partition[i] = 2;
        else
            partition[i] = 3;
    }
    double marginalThreeParts = computeLogMarginalLikelihood(templateImage, partition, vectorFields, variance, priorPrecision);
    std::cout << "marginal three parts = " << marginalThreeParts << std::endl;
    double bayesFactorThree = std::exp(marginalThreeParts - marginalTwoParts);
    std::cout << "bayes factor, std::exp(marginalThreeParts - marginalTwoParts) = " << bayesFactorThree << std::endl;
    
    // test log gamma in 3 dimensions
    std::cout << "LogGamma3(5) = " << LogGamma3(5) << std::endl;
    std::cout << "In R: pi^(3/2) * gamma(5) * gamma(5-1/2) * gamma(5-1) = 9.140645" << std::endl;
    
//    vnl_matrix< double > K = variance*B;
//    vnl_matrix< double > Sigma = variance*idMatrix;
//    std::cout << "K %x% solve(Sigma) = \n" << KroneckerProduct(K,vnl_cholesky(Sigma).inverse()) << std::endl;
//    double separateMarginalUppercase = computeLogMarginalLikelihoodUppercaseSigma(partitionImage, vectorFields, K, Sigma);
//    std::cout << "separate marginal uppercase Sigma = " << separateMarginalUppercase << std::endl;
//    double jointMarginalUppercase = computeLogMarginalLikelihoodUppercaseSigma(partitionImage, vectorFields, K, Sigma);
//    std::cout << "joint marginal uppercase Sigma = " << jointMarginalUppercase << std::endl;
//    double bayesFactorUppercase = std::exp(jointMarginalUppercase - separateMarginalUppercase);
//    std::cout << "bayes factor uppercase Sigma = " << bayesFactorUppercase << std::endl;    
}

int main( int argc, char** argv ) {
    
    boost::mt19937 gen;
    
    //itk::MultiThreader::SetGlobalDefaultNumberOfThreads(1);
    
    std::string templateImageName;
    std::string vectorFieldDirName;
    double variance;
    double alpha;
    unsigned int axis;
    unsigned int slice;
    unsigned int visualExpandFactor;
    unsigned int shrinkFactor;
    unsigned int noOfSteps;
    
    // define hyperparameter
    vnl_matrix< double > B(Dimension+1,Dimension+1);
    B.fill(0.0);
    B(0,0) = 100;
    B(1,1) = 100;
    B(2,2) = 100;
    B(3,3) = 1;
    vnl_matrix< double > idMatrix(Dimension,Dimension);
    idMatrix.set_identity();
    vnl_matrix< double > priorPrecision = KroneckerProduct(B, idMatrix);
    // lower scaling
//    priorPrecision(0,0) = 1e+5;
//    priorPrecision(4,4) = 1e+5;
//    priorPrecision(8,8) = 1e+5;
    // lower rotation
//    priorPrecision(1,1) = 1e+5;
//    priorPrecision(2,2) = 1e+5;
//    priorPrecision(3,3) = 1e+5;
//    priorPrecision(5,5) = 1e+5;
//    priorPrecision(6,6) = 1e+5;
//    priorPrecision(7,7) = 1e+5;
    std::cout << "priorPrecision = \n" << priorPrecision << std::endl;

	if(argc == 1) {
                
        // create test case
        std::cout << "Start test case: Two translations" << std::endl;
        templateImageName = "TestTemplate.mha";
        vectorFieldDirName = "./fields";
        itk::FileTools::CreateDirectory(vectorFieldDirName.c_str());
        unsigned int sizeX = 10, sizeY = 10, sizeZ = 20;
        variance = 1e-1;
        double noise = 1e-2;
        testCase(templateImageName, vectorFieldDirName, sizeX, sizeY, sizeZ, variance, noise, priorPrecision);
        alpha = 1e-5;
        axis = 1;
        slice = sizeY/2;
        visualExpandFactor = sizeX*10;
        shrinkFactor = 1;
        noOfSteps = 20;
	}
    else if(argc == 7) {
        
        templateImageName = argv[1];
        std::cout << "templateImageName = " << templateImageName << std::endl;
        std::string vectorFieldFileName = argv[2];
        std::cout << "vectorFieldFileName = " << vectorFieldFileName << std::endl;
        variance = atof(argv[3]);
        std::cout << "variance = " << variance << std::endl;
        shrinkFactor = atoi(argv[4]);
        std::cout << "shrinkFactor = " << shrinkFactor << std::endl;
        std::string partitionImageName1 = argv[5];
        std::cout << "partitionImageName1 = " << partitionImageName1 << std::endl;
        std::string partitionImageName2 = argv[6];
        std::cout << "partitionImageName2 = " << partitionImageName2 << std::endl;
        
        // template
        typedef itk::ImageFileReader< ImageType > ImageReaderType;
        ImageReaderType::Pointer imageReader = ImageReaderType::New();
        imageReader->SetFileName(templateImageName);
        imageReader->Update();
        ImageType::Pointer templateImage = imageReader->GetOutput();
        templateImage->DisconnectPipeline();
        
        typedef itk::ShrinkImageFilter< ImageType,ImageType > ShrinkImageFilterType;
        ShrinkImageFilterType::Pointer shrinkFilter = ShrinkImageFilterType::New();
        shrinkFilter->SetInput(templateImage);
        shrinkFilter->SetShrinkFactors(shrinkFactor);
        shrinkFilter->UpdateLargestPossibleRegion();
        templateImage = shrinkFilter->GetOutput();
        templateImage->DisconnectPipeline();
                
        ImageType::SizeType templateSize = templateImage->GetLargestPossibleRegion().GetSize();
        unsigned noOfVoxels = templateSize[0]*templateSize[1]*templateSize[2];
        
        // partition 1
        typedef itk::ImageFileReader< LabelImageType > LabelImageReaderType;
        LabelImageReaderType::Pointer labelImageReader = LabelImageReaderType::New();
        labelImageReader->SetFileName(partitionImageName1);
        labelImageReader->Update();
        LabelImageType::Pointer partitionImage1 = labelImageReader->GetOutput();
        partitionImage1->DisconnectPipeline();

        std::vector< LabelPixelType > partition1(noOfVoxels);
        itk::ImageRegionIterator< LabelImageType > partitionIter1(partitionImage1, partitionImage1->GetLargestPossibleRegion());
        unsigned id = 0;
        for(partitionIter1.GoToBegin(); !partitionIter1.IsAtEnd(); ++partitionIter1) {
            partition1[id] = partitionIter1.Get();
            ++id;
        }

        // partition 2
        labelImageReader->SetFileName(partitionImageName2);
        labelImageReader->Update();
        LabelImageType::Pointer partitionImage2 = labelImageReader->GetOutput();
        partitionImage2->DisconnectPipeline();
        
        std::vector< LabelPixelType > partition2(noOfVoxels);
        itk::ImageRegionIterator< LabelImageType > partitionIter2(partitionImage2, partitionImage2->GetLargestPossibleRegion());
        id = 0;
        for(partitionIter2.GoToBegin(); !partitionIter2.IsAtEnd(); ++partitionIter2) {
            partition2[id] = partitionIter2.Get();
            ++id;
        }
        
        // velocity field
        typedef itk::ImageFileReader< VectorFieldType > VectorFieldReaderType;
        VectorFieldReaderType::Pointer fieldReader = VectorFieldReaderType::New();
        fieldReader->SetFileName(vectorFieldFileName);
        fieldReader->Update();
        std::vector< VectorFieldType::Pointer > vectorFields(1);
        vectorFields[0] = fieldReader->GetOutput();
        vectorFields[0]->DisconnectPipeline();
        
        typedef itk::ShrinkImageFilter< VectorFieldType,VectorFieldType > ShrinkImageFilterFieldType;
        ShrinkImageFilterFieldType::Pointer shrinkFieldFilter = ShrinkImageFilterFieldType::New();
        shrinkFieldFilter->SetInput(vectorFields[0]);
        shrinkFieldFilter->SetShrinkFactors(shrinkFactor);
        shrinkFieldFilter->UpdateLargestPossibleRegion();
        vectorFields[0] = shrinkFieldFilter->GetOutput();
        vectorFields[0]->DisconnectPipeline();

        // Bayes factor
        double ml1 = computeLogMarginalLikelihood(templateImage, partition1, vectorFields, variance, priorPrecision);
        std::cout << "marginal likelihood 1 = " << ml1 << std::endl;
        double ml2 = computeLogMarginalLikelihood(templateImage, partition2, vectorFields, variance, priorPrecision);
        std::cout << "marginal likelihood 2 = " << ml2 << std::endl;
        std::cout << "Bayes factor = " << std::exp(ml1-ml2) << std::endl;
        
        return EXIT_SUCCESS;
    }
    else if(argc == 10) {
        templateImageName = argv[1];
        std::cout << "templateImageName = " << templateImageName << std::endl;
        vectorFieldDirName = argv[2];
        std::cout << "vectorFieldDirName = " << vectorFieldDirName << std::endl;
        variance = atof(argv[3]);
        std::cout << "variance = " << variance << std::endl;
        alpha = atof(argv[4]);
        std::cout << "alpha = " << alpha << std::endl;
        axis = atoi(argv[5]);
        std::cout << "axis = " << axis << std::endl;
        slice = atoi(argv[6]);
        std::cout << "slice = " << slice << std::endl;
        visualExpandFactor = atoi(argv[7]);
        std::cout << "visualExpandFactor = " << visualExpandFactor << std::endl;
        shrinkFactor = atoi(argv[8]);
        std::cout << "shrinkFactor = " << shrinkFactor << std::endl;
        noOfSteps = atoi(argv[9]);
        std::cout << "noOfSteps = " << noOfSteps << std::endl;
    }
    else {
        std::cout << "Usage: BayesianNonparameterics TemplateImage VectorFieldDir VelocityVariance ClusterAlpha PNGAxis PNGSlice PNGExpandFactors ShrinkFactor NoOfSteps"
        << std::endl;
        return EXIT_FAILURE;
    }
    
    const char* partitionMassFilename = "PartitionMass.txt";
    
	typedef itk::ImageFileReader< ImageType > ImageReaderType;
    ImageReaderType::Pointer imageReader = ImageReaderType::New();
	imageReader->SetFileName(templateImageName);
	imageReader->Update();
    ImageType::Pointer templateImage = imageReader->GetOutput();
    templateImage->DisconnectPipeline();
    
    typedef itk::ShrinkImageFilter< ImageType,ImageType > ShrinkImageFilterType;
    ShrinkImageFilterType::Pointer shrinkFilter = ShrinkImageFilterType::New();
    shrinkFilter->SetInput(templateImage);
    shrinkFilter->SetShrinkFactors(shrinkFactor);
    shrinkFilter->UpdateLargestPossibleRegion();
    templateImage = shrinkFilter->GetOutput();
    templateImage->DisconnectPipeline();
    
    itk::ImageFileWriter< ImageType >::Pointer imageWriter = itk::ImageFileWriter< ImageType >::New();
    imageWriter->SetInput(templateImage);
    imageWriter->SetFileName("TemplateImage_Resampled.mha");
    imageWriter->Update();
    
//    // test id to index
//    ImageType::IndexType testIndex = {{2,7,4}};
//    ImageType::SizeType testSize = {{15,10,20}};
//    unsigned int testId = indexToId(testIndex, testSize);
//    ImageType::IndexType returnTestIndex = idToIndex(testId, testSize);
//    std::cout << "returnTestIndex = " << returnTestIndex << std::endl;
    
    // f(d_nm) = 1 voxel
    ImageType::SizeType templateSize = templateImage->GetLargestPossibleRegion().GetSize();
    unsigned noOfVoxels = templateSize[0]*templateSize[1]*templateSize[2];
    vnl_sparse_matrix< float > distanceMatrix(noOfVoxels,noOfVoxels);
    itk::ImageRegionConstIteratorWithOnlyIndex< ImageType > templateIter(templateImage, templateImage->GetLargestPossibleRegion());
    for(templateIter.GoToBegin(); !templateIter.IsAtEnd(); ++templateIter) {
        
        ImageType::IndexType currentIndex = templateIter.GetIndex();
        unsigned int row = indexToId(currentIndex,templateSize);
//        std::cout << "index = " << currentIndex << " id = " << row << std::endl;
        
        for(unsigned int j = 0; j < NoOfNeighborsWithinDistance1; ++j) {
            
            ImageType::IndexType neighborIndex = currentIndex;
            for(unsigned int i = 0; i < Dimension; ++i)
                neighborIndex[i] = neighborIndex[i] + offset[j][i];
            
            if(templateImage->GetLargestPossibleRegion().IsInside(neighborIndex)) {
                unsigned int col = indexToId(neighborIndex,templateSize);
                distanceMatrix.put(row,col,1);
            }
        }
        
        // center
        distanceMatrix.put(row,row,alpha);
    }
    
//    std::cout << "distanceMatrix.rows() = " << distanceMatrix.rows() << std::endl;
//    for(unsigned int i = 0; i < distanceMatrix.rows(); ++i) {
//        vnl_sparse_matrix< float >::row pairs = distanceMatrix.get_row(i);
//        std::cout << "number of pairs = " << pairs.size() << std::endl;
//        for(unsigned int j = 0; j < pairs.size(); ++j)
//            std::cout << pairs[j].first << " " << pairs[j].second << std::endl;
//    }

    // create initial link structure
    std::cout << "create initial link structure" << std::endl;
    std::vector< unsigned int > links(noOfVoxels);
    
    // random partition from ddCPR
    for(unsigned int i = 0; i < links.size(); ++i) {

        vnl_sparse_matrix< float >::row pairs = distanceMatrix.get_row(i);

        std::vector< double > weights(pairs.size());
        for(unsigned int j = 0; j < pairs.size(); ++j)
            weights[j] = pairs[j].second;
        
        boost::random::discrete_distribution<> rdiscrete(weights);
        int pick = rdiscrete(gen);

        links[i] = pairs[pick].first;

    }
    
//    // one partition per voxel
//    for(unsigned int i = 0; i < noOfVoxels; ++i)
//        links[i] = i;
    
    // deterministically obtain partitions from link structure
    std::cout << "deterministically obtain partitions from link structure" << std::endl;
    std::vector< LabelPixelType > partition(links.size());
    for(unsigned int i = 0; i < partition.size(); ++i)
        partition[i] = 0;
    createPartition(links, partition);
    
    // create new file
    std::ofstream massFile;
    massFile.open(partitionMassFilename);
//    // write intitial partitioning
//    massFile << writePartitionStatistics(partitionImage);
    unsigned int noOfPartitions = *std::max_element(partition.begin(), partition.end());
    std::cout << "noOfPartitions = " << noOfPartitions << std::endl;
    massFile << noOfPartitions << std::endl;

    unsigned int step = 0;
    writePartition(templateImage, partition, axis, slice, visualExpandFactor, step, shrinkFactor);
    
    // load all vector fields
    itk::RegularExpressionSeriesFileNames::Pointer directory = itk::RegularExpressionSeriesFileNames::New();
    directory->SetDirectory(vectorFieldDirName.c_str());
    directory->SetRegularExpression(".mha");
    std::vector< std::string > vectorFieldNames = directory->GetFileNames();
    std::vector< VectorFieldType::Pointer > vectorFields(vectorFieldNames.size());
    
    // separate itk::Image in a std::vector
    for(unsigned int i = 0; i < vectorFieldNames.size(); ++i) {
        
        std::cout << i << " " << vectorFieldNames[i] << std::endl;
        typedef itk::ImageFileReader< VectorFieldType > VectorFieldReaderType;
        VectorFieldReaderType::Pointer fieldReader = VectorFieldReaderType::New();
        fieldReader->SetFileName(vectorFieldNames[i]);
        fieldReader->Update();
        vectorFields[i] = fieldReader->GetOutput();
        vectorFields[i]->DisconnectPipeline();
        
        typedef itk::ShrinkImageFilter< VectorFieldType,VectorFieldType > ShrinkImageFilterType;
        ShrinkImageFilterType::Pointer shrinkFilter = ShrinkImageFilterType::New();
        shrinkFilter->SetInput(vectorFields[i]);
        shrinkFilter->SetShrinkFactors(shrinkFactor);
        shrinkFilter->UpdateLargestPossibleRegion();
        vectorFields[i] = shrinkFilter->GetOutput();
        vectorFields[i]->DisconnectPipeline();
        
        itk::ImageFileWriter< VectorFieldType >::Pointer fieldImageWriter = itk::ImageFileWriter< VectorFieldType >::New();
        std::ostringstream fieldFilename;
        fieldFilename << "VectorField_" << i << ".mha";
        fieldImageWriter->SetFileName(fieldFilename.str());
        fieldImageWriter->SetInput(vectorFields[i]);
        fieldImageWriter->UseCompressionOn();
        fieldImageWriter->Update();

    }
        
    // Gibbs sampler
    for(step = 1; step < noOfSteps; ++step) {
        
        std::cout << "------------- Gibbs step: " << step << " -------------" << std::endl;
        
        for(unsigned int i = 0; i < links.size(); ++i) {            
            if(i % 100 == 0)
                std::cout << "link: " << i << std::endl;
            
            // self link
            links[i] = i;
            
            // deterministically obtain partitions from link structure
            createPartition(links, partition);
            // fast version
            LabelImageType::PixelType part1 = partition[links[i]];
            double part1Marginal = computeLogMarginalLikelihood(templateImage, partition, vectorFields, variance, priorPrecision, part1);
            
//            // slow version: compute all marginal likelihoods
//            double oldMarginal = computeLogMarginalLikelihood(templateImage, partition, vectorFields, variance, priorPrecision);
            
            vnl_sparse_matrix< float >::row pairs = distanceMatrix.get_row(i);
            std::vector< double > weights(pairs.size());
            for(unsigned int j = 0; j < pairs.size(); ++j) {
                    
                if(pairs[j].first == i) {
                    weights[j] = pairs[j].second;
                }
                else {
                    // fast version
                    double bayesFactor = 1;
                    LabelImageType::PixelType part2 = partition[pairs[j].first];
                    
                    if(part1 != part2) {
                        double part2Marginal = computeLogMarginalLikelihood(templateImage, partition, vectorFields, variance, priorPrecision, part2);

                        std::vector< unsigned int > linksTryout = links;
                        std::vector< LabelPixelType > partitionTryout = partition;

                        linksTryout[i] = pairs[j].first;
                        createPartition(linksTryout, partitionTryout);
                        LabelImageType::PixelType jointParts = partitionTryout[linksTryout[i]];
                        double jointMarginal = computeLogMarginalLikelihood(templateImage, partitionTryout, vectorFields, variance, priorPrecision, jointParts);
                        
                        // Bayes factor for model selection
                        bayesFactor = std::exp(jointMarginal - part1Marginal - part2Marginal);
                    }
                    
//                    // slow version: compute all marginal likelihoods
//                    links[i] = pairs[j].first;
//                    createPartition(links, partition);
//                    double newMarginal = computeLogMarginalLikelihood(templateImage, partition, vectorFields, variance, priorPrecision);
//                    double bayesFactor = std::exp(newMarginal - oldMarginal);
                    
                    // adjust the probabilities with the Bayes factor
                    weights[j] = pairs[j].second * bayesFactor;
                    
                }
            }
            
            boost::random::discrete_distribution<> rdiscrete(weights);
            int pick = rdiscrete(gen);
            links[i] = pairs[pick].first;
//            std::cout << "weights = ";
//            for(unsigned int k = 0; k < weights.size(); ++k)
//                std::cout << weights[k] << " ";
//            std::cout << " pick = " << pick << std::endl;
            
        }
        
        createPartition(links, partition);
        
//        massFile << writePartitionStatistics(partitionImage);
        unsigned int noOfPartitions = *std::max_element(partition.begin(), partition.end());
        std::cout << "noOfPartitions = " << noOfPartitions << std::endl;
        std::cout << "marginal likelihood = " << computeLogMarginalLikelihood(templateImage, partition, vectorFields, variance, priorPrecision)  << std::endl;
        massFile << noOfPartitions << std::endl;
        
        writePartition(templateImage, partition, axis, slice, visualExpandFactor, step, shrinkFactor);

    }
    
    massFile.close();
    
	return EXIT_SUCCESS;
}
