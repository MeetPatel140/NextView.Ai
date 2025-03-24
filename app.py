from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/datasets/<int:id>', methods=['DELETE'])
def delete_dataset(id):
    # Logic to delete the dataset with the given id
    # For example, remove the dataset from the database
    # dataset = get_dataset_by_id(id)
    # if dataset:
    #     delete_dataset_from_db(id)
    #     return jsonify({'message': 'Dataset deleted successfully'}), 200
    # else:
    #     return jsonify({'error': 'Dataset not found'}), 404
    return jsonify({'message': 'Dataset deleted successfully'}), 200

if __name__ == '__main__':
    app.run(debug=True)