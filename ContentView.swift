import SwiftUI
import UIKit
import Foundation

struct ContentView: View {
    @State private var image: UIImage?
    @State private var showImagePicker = false
    @State private var isCamera = false
    @State private var predictionResult: String = ""
    @State private var predictionScore: Double = 0.0
    
    let gradientColors = [Color.white, Color(red: 0.2, green: 0.2, blue: 0.2)]
    
    var body: some View {
        ZStack {
            LinearGradient(gradient: Gradient(colors: gradientColors),
                        startPoint: .top, endPoint: .bottom)
                .edgesIgnoringSafeArea(.all)
            
            VStack {
                
            if let image = image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(width: 300, height: 300)
            } else {
                Text("Select an image")
                    .foregroundColor(.gray)
                    .font(.headline)
                    .padding(.top, 50)
            }
            
            Spacer()
            
            if !predictionResult.isEmpty {
                Text("Prediction: \(predictionResult)")
                        .font(.headline)
                        .padding()
                        .foregroundColor(.white)
                Text("Score: \(predictionScore)")
                        .font(.subheadline)
                        .padding()
                        .foregroundColor(.white)
            } else {
                Text("Prediction: NONE")
                    .font(.headline)
                    .padding()
                    .foregroundColor(.white)
                Text("Score: NONE")
                    .font(.subheadline)
                    .padding()
                    .foregroundColor(.white)
            }
        }
            
            Spacer()
            
            VStack(spacing: 20) {
                HStack(spacing: 40) {
                    VStack(spacing: 1) {
                        
                        Image("Camera")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 60, height: 60)
                        
                        Button("Take Photo") {
                            isCamera = true
                            showImagePicker = true
                        }
                        .foregroundColor(.white)
                    }
                    
                    VStack(spacing: 10) {
                        Image("Library")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 60, height: 60)
                        
                        Button("Choose from Library") {
                            isCamera = false
                            showImagePicker = true
                        }
                        .foregroundColor(.white)
                    }
                }
                
                VStack(spacing: 10) {
                    Image("Trash can")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 60, height: 60)
                    
                    Button("Scan trash") {
                        if let image = image {
                            uploadImageToServer(image: image)
                        }
                    }
                    .foregroundColor(.white)
                }
            }
            .padding()
            
        }
        .sheet(isPresented: $showImagePicker) {
            ImagePicker(sourceType: isCamera ? .camera : .photoLibrary, selectedImage: $image)
        }
    }
    
    func uploadImageToServer(image: UIImage) {
        guard let url = URL(string: "https://8c77-2601-88-8100-cd60-98b3-f8a0-7f1d-db8f.ngrok-free.app/upload") else { return }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        let imageData = image.jpegData(compressionQuality: 0.8)!
        let fieldName = "file"
        let fileName = "image.jpg"
        
        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"\(fieldName)\"; filename=\"\(fileName)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("Error: \(error.localizedDescription)")
                return
            }
            
            guard let data = data else { return }
            
            do {
                if let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] {
                    DispatchQueue.main.async {
                        if let success = json["success"] as? Bool, success {
                            self.predictionResult = json["results"] as? String ?? "Unknown"
                            self.predictionScore = json["score"] as? Double ?? 0.0
                        } else {
                            print("Error in response: \(json)")
                        }
                    }
                }
            } catch {
                print("Error parsing JSON: \(error.localizedDescription)")
            }
        }.resume()
    }
}

struct ImagePicker: UIViewControllerRepresentable {
    var sourceType: UIImagePickerController.SourceType
    @Binding var selectedImage: UIImage?
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = sourceType
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        var parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let image = info[.originalImage] as? UIImage {
                parent.selectedImage = image
            }
            picker.dismiss(animated: true)
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true)
        }
    }
}
